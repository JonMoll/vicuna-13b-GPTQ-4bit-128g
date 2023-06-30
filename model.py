import torch
import transformers

from transformers import AutoConfig
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from gptq_llama.quant import make_quant
from gptq_llama.modelutils import find_layers

from utils import Stream
from utils import Iteratorize
from utils import clear_torch_cache
from utils import _SentinelTokenStoppingCriteria


class LLMVicuna:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self,
                   path_model,
                   name_checkpoint,
                   wbits=4,
                   groupsize=128,
                   faster_kernel=False,
                   kernel_switch_threshold=128,
                   model_seqlen=2048):

        skip_function = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = skip_function
        torch.nn.init.uniform_ = skip_function
        torch.nn.init.normal_ = skip_function

        config = AutoConfig.from_pretrained(path_model)
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        self.model = AutoModelForCausalLM.from_config(config)
        torch.set_default_dtype(torch.float)
        self.model = self.model.eval()

        layers = find_layers(self.model)
        del layers['lm_head']

        make_quant(self.model, layers, wbits, groupsize, faster_kernel, '', kernel_switch_threshold)
        del layers

        path_checkpoint = f'{path_model}/{name_checkpoint}'
        self.model.load_state_dict(load_file(path_checkpoint), strict=False)
        self.model.seqlen = model_seqlen
        self.model = self.model.to(torch.device('cuda:0'))

        self.tokenizer = LlamaTokenizer.from_pretrained(path_model, clean_up_tokenization_spaces=True)
        self.tokenizer.truncation_side = 'left'

    def encode(self, prompt, tokens_to_generate=0, add_special_tokens=True):
        max_length = 2048 - tokens_to_generate
        input_ids = self.tokenizer.encode(str(prompt),
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=max_length,
                                          add_special_tokens=add_special_tokens)

        if input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        return input_ids.cuda()

    def decode(self, output_ids):
        reply = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply

    def generate_with_callback(self, callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_with_streaming(self, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs, callback=None)

    def generate_reply(self,
                       prompt,
                       stopping_strings=['\nYou:', '\nAssistant:'],
                       max_new_tokens=200,
                       do_sample=True,
                       temperature=1.99,
                       top_p=0.18,
                       typical_p=1.0,
                       repetition_penalty=1.15,
                       encoder_repetition_penalty=1,
                       top_k=30,
                       min_length=0,
                       no_repeat_ngram_size=0,
                       num_beams=1,
                       penalty_alpha=0,
                       length_penalty=1,
                       early_stopping=False):

        clear_torch_cache()

        input_ids = self.encode(prompt, max_new_tokens)

        eos_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        stopping_criteria_list = transformers.StoppingCriteriaList()

        t = [self.encode(string, 0, add_special_tokens=False) for string in stopping_strings]
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=t, starting_idx=len(input_ids[0])))

        generate_params = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_p': top_p,
            'typical_p': typical_p,
            'repetition_penalty': repetition_penalty,
            'encoder_repetition_penalty': encoder_repetition_penalty,
            'top_k': top_k,
            'min_length': min_length,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'num_beams': num_beams,
            'penalty_alpha': penalty_alpha,
            'length_penalty': length_penalty,
            'early_stopping': early_stopping,
            'eos_token_id': eos_token_ids,
            'stopping_criteria': stopping_criteria_list,
            'inputs': input_ids,
        }

        with self.generate_with_streaming(**generate_params) as generator:
            for output in generator:
                new_tokens = len(output) - len(input_ids[0])
                reply = self.decode(output[-new_tokens:])
                yield reply

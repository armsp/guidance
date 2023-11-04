from .._transformers import Transformers

class LLaMA(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # load the LLaMA specific tokenizer and model
        import transformers
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model, **kwargs)
            model = transformers.LlamaForCausalLM.from_pretrained(model, **kwargs)
            
        return super()._model_and_tokenizer(model, tokenizer, **kwargs)
    
class LLaMAChat(LLaMA):

    default_system_prompt = """- You are a helpful assistant chatbot.  
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes."""

    @staticmethod
    def role_start(role):
        return  {
            'user': '<|im_start|>user ',
            'system': '<|im_start|>system\n',
            'assistant': '<|im_start|>assistant ',
            }[role]

    @staticmethod
    def role_end(role):
        return '<|im_end|>'

import functools
import os
from typing import Optional

from transformers import AutoTokenizer
from stickytoken.utils import write_tokenizer_analysis_results


def model_needs_fast_tokenizer(model_id):
    return (  # slow tokenizer not provided or has issues
        "Cohere" in model_id
        or "stabilityai" in model_id
        or "gpt-j" in model_id
        or "pythia" in model_id
        or "neox" in model_id
        or "OLMo" in model_id
    )


class TokenizerAnalyzer:
    START_PREFIX = "«"
    SPACE_CHARS = " ▁_Ġ"
    FORCE_STARTING_SPACE_MODELS = [
        "01-ai/Yi",
        "ai21labs/Jamba-v0.1",
    ]  # does not add, but still removes?

    def __init__(
        self,
        model_id: str,
        use_fast: bool = None,
        trust_remote_code: bool = False,
        tokenizer=None,
    ):
        if use_fast is None:
            use_fast = model_needs_fast_tokenizer(model_id)
        self.model_name =  os.path.basename(model_id)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_id,
            use_fast=use_fast,
            clean_up_tokenization_spaces=False,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            print(
                f"Warning: The tokenizer for {model_id} does not have pad_token_id, setting it to eos_token_id = {self.tokenizer.eos_token_id}"
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model_id = model_id
        self.vocab_s2i = self.tokenizer.get_vocab() # dict[str, int]
        self.vocab_i2s = {v: k for k, v in self.vocab_s2i.items()} # dict[int, str]

        self.special_token_ids = getattr(self.tokenizer, "additional_special_tokens_ids") or []
        for attr in [
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "sep_token_id",
            "mask_token_id",
        ]:
            token_id = getattr(self.tokenizer, attr)
            if token_id is not None:
                self.special_token_ids.append(token_id)

        self._set_flags()

    def _set_flags(self):
        # check if the tokenizer uses GPT-2 style byte encoding
        self.gpt2_style_byte_encoding = (
            sum(v.startswith("Ġ") for v in self.vocab_i2s.values()) > len(self.vocab_i2s) / 10
        )

        # we can not rely on huggingface settings like add_prefix_space or cleanup_tokenization_spaces
        # as some models use preprocessors etc instead
        # so, we check if the input appears manipulated, and counteract it if necessary
        prefix_str = self.START_PREFIX # "«"
        test_phrase_without_space = "test"
        tokens_no_space = self.tokenizer.encode(test_phrase_without_space, add_special_tokens=False)
        print(f"Testing {self.model_name=} with {test_phrase_without_space!r} -> {tokens_no_space}")

        if self.vocab_i2s[tokens_no_space[0]][0] in self.SPACE_CHARS or any(
            p in self.model_id for p in self.FORCE_STARTING_SPACE_MODELS
        ):
            enc_prefix = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            assert len(enc_prefix) == 1, f"The prefix should be a single token, but was {enc_prefix}"
            self.starting_space_mode = True
            self.start_prefix_id = enc_prefix[0]
            print(
                f"vocab[{tokens_no_space[0]}] = {self.vocab_i2s[tokens_no_space[0]]}"
            )
            print(
                f"Warning: The tokenizer for {self.model_name} adds spaces to the start or does other space manipulations, trying to counteract it by using the prefix {prefix_str!r} = {enc_prefix}"
            )
        else:
            assert (
                self.vocab_i2s[tokens_no_space[0]][0] == "t"
            ), f"The first character of the first token should be 't' for {test_phrase_without_space!r} but found {self.vocab_i2s[tokens_no_space[0]]}"
            self.starting_space_mode = False
            self.start_prefix_id = None
        print(f"Starting space mode: {self.starting_space_mode}")

    def clean_encode(self, s) -> list[int]:
        '''
        sen_t5:
        toka.clean_decode([794]),toka.clean_decode([4377])
        >>(' test', 'test')
        toka.clean_encode(toka.clean_decode([794])),toka.clean_encode(toka.clean_decode([4377]))
        >>([794], [4377])
        '''
        if self.starting_space_mode:
            s = self.START_PREFIX + s #' test' >>  '« test'
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            assert (
                tokens[0] == self.start_prefix_id
            ), f"The first token should be the prefix {self.start_prefix_id} for {s!r} but found {tokens}"
            tokens = tokens[1:]
        else:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
        return tokens

    def clean_decode(self, tokens):
        '''
        sen_t5:
        toka.clean_decode([794]),toka.clean_decode([4377])
        >>(' test', 'test')
        '''
        if self.starting_space_mode:  # this prevents the tokenizer from dropping the starting space when it exists
            tokens = [self.start_prefix_id] + tokens
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            if decoded[0] == " ":  # e.g. mistral, but not llama2
                decoded = decoded[1:]
            assert decoded.startswith(
                self.START_PREFIX
            ), f"The decoded string {decoded!r} should start with the prefix for {tokens!r}"
            return decoded[1:]
        else:
            return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def find_substring_tokens(self, token_id: int) -> list:
        return [(i, s) for i, s in self.vocab_i2s.items() if self.vocab_i2s[token_id] in s and i != token_id]

    def categorize_token(self, token_id: int) -> dict:
        """Categorize a token based on its encoding and decoding behavior."""
        s = self.clean_decode([token_id]) # str
        tokens = self.clean_encode(s)
        if s.strip() == "":
            category = "MEANINGLESS"
        elif tokens == [token_id]:
            category = "OK_SPECIAL" if token_id in self.special_token_ids else "OK"
        elif len(s) >= 3 and s[0] in "[<" and s[-1] in "]>" and any(c.isalpha() for c in s):
            category = "UNREACHABLE_SPECIAL"  # [BOS], </s> and the like
        elif "�" in s:
            category = "UNDECODEABLE"
        elif len(tokens) == 1:
            category = "UNREACHABLE_SINGLE_TOKEN"
        else:  #  len(tokens) > 1:
            category = "UNREACHABLE_MULTI_TOKEN"
        
        token_info = dict(i=token_id, raw_vocab=self.vocab_i2s[token_id], category=category, decoded=s)
        if tokens != [token_id] and "�" not in s:
            token_info["reencoded_ids"] = tokens
            token_info["reencoded"] = [self.vocab_to_readable_string(t) for t in tokens]

        return token_info

    def categorize_tokens(self) -> dict[int, dict]:
        """Categorize tokens into different categories based on their encoding and decoding behavior."""
        vocab_infos =  {i: self.categorize_token(i) for i in self.vocab_i2s}
        write_tokenizer_analysis_results(vocab_infos, self.model_name, compress=False)
        return vocab_infos


    def vocab_to_readable_string(self, token_id: int) -> str:
        """essentially performs manual and slow UTF-8 decoding to handle stray bytes"""
        s = self.vocab_i2s[token_id]
        if self.gpt2_style_byte_encoding:
            m = gpt2_vocab_to_bytes()
            try:
                bs = [m[c] for c in s]
            except KeyError:
                return f"¿{s}?"  # happens on misconfigured tokenizers
            decoded_chars = []
            i = 0
            while i < len(bs):
                nb = _utf_byte_type(bs[i])
                if nb == 0 or nb == 5:
                    decoded_chars.append(_hexbyte(bs[i]))  # 0 should only happen at start, 5 only invalid escape
                    i += 1
                elif nb == 1:
                    decoded_chars.append(chr(bs[i]))
                    i += 1
                else:
                    if i + nb > len(bs):
                        break
                    decoded_chars.append(bytes(bs[i : i + nb]).decode("utf-8"))
                    i += nb
            while i < len(bs):
                decoded_chars.append(_hexbyte(bs[i]))
                i += 1
            return "".join(decoded_chars)
        else:  # some already use valid strings without stray bytes
            if s.startswith("▁"):
                s = " " + s[1:]
            return s

    def token_byte_value(self, token_id: int) -> Optional[int]:
        """Return the byte value of a token if it is a single byte character
        - ascii 0-127
        - fallback <0xAB> or single character in the gpt2 encoding
        - but NOT unicode 128-255, which should encode to two bytes in utf-8
        if not one of these, returns None."""
        if self.gpt2_style_byte_encoding:
            if len(self.vocab_i2s[token_id]) != 1:
                return None
            s = self.vocab_to_readable_string(token_id)
        else:
            s = self.vocab_i2s[token_id]
        if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):  # <0xFF>
            return int(s[3:5], 16)
        elif len(s) != 1 or ord(s) > 127:
            return None
        else:
            return ord(s)


@functools.cache
def gpt2_bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings as done by gpt2 and many others."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@functools.cache
def gpt2_vocab_to_bytes():
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


@functools.cache
def _utf_byte_type(b: int):
    start_byte = f"{b:08b}"  # cached so we can be really explicit
    if start_byte.startswith("10"):  # continuation byte
        return 0
    if start_byte.startswith("0"):
        return 1
    if start_byte.startswith("110"):
        return 2
    if start_byte.startswith("1110"):
        return 3
    if start_byte.startswith("11110"):
        return 4
    return 5  # not part of utf8


@functools.cache
def _hexbyte(b):
    return f"<0x{b:02X}>"

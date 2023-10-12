
# Load in libraries
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sn
from cryptography.fernet import Fernet
import argparse
from pathlib import Path
import json
from importlib import import_module
from cryptography.fernet import Fernet
from tqdm import tqdm

# Local functions
def get_args():
    """Get's the arguments needed for running the script:
    -i, --inputmodel: absolute path to model-folder <Path>
    -m, --message: message to be encrypted/encoded in the model <string>
    -s, --savemodel: flag to save the model <Bool>
    -r, --readmodel: flag to read the hacked model with cipher/key <Bool>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputmodel", required=True)
    parser.add_argument("-m", "--message", required=False)
    parser.add_argument("-s", "--savemodel", required=False, action='store_true')
    parser.add_argument("-r", "--readmodel", required=False, action='store_true')
    args = parser.parse_args()
    if args.readmodel != False:
        input_dir = Path(args.inputmodel)
        readmodel = args.readmodel
        if not input_dir.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)
        return input_dir, None, False, readmodel
    else:
        if args.savemodel != False:
            input_dir = Path(args.inputmodel)
            message = args.message
            savemodel = args.savemodel
            if not input_dir.exists():
                print("The target directory doesn't exist")
                raise SystemExit(1)
            return input_dir, message, savemodel, False
        else:
            input_dir = Path(args.inputmodel)
            message = args.message
            if not input_dir.exists():
                print("The target directory doesn't exist")
                raise SystemExit(1)
            return input_dir, message, False, False
        
def get_model_configs(input_dir):
    """Gets the model configuration names needed to initialize the model and 
    the tokenizer
    """
    print(f"#0: Loading model configs")
    model_config_data = {}
    for entry in input_dir.iterdir():
        if entry.name == "config.json":
            with open(entry, 'r') as f:
                model_config_data = json.loads(f.read())
    model_arch = model_config_data['architectures'][0]
    model_type = model_config_data['model_type']
    tokenizer_arch = f"{model_arch[:len(model_type)]}Tokenizer"
    return model_arch, model_type, tokenizer_arch

def load_trf_modules(model_arch, tokenizer_arch, input_dir):
    """Initializes the correct model and tokenizer objects
    """
    loaded_model_arch = getattr(import_module('transformers'), model_arch)
    loaded_tokenizer_arch = getattr(import_module('transformers'), tokenizer_arch)
    t_model = loaded_model_arch.from_pretrained(input_dir)
    tokenizer = loaded_tokenizer_arch.from_pretrained(input_dir)
    return t_model, tokenizer

def gen_char_list(message):
    """Generates character list based on the input-message string
    """
    # print(f"Original message: {message}")
    char_list = [char for word in message for char in word]
    # print(f"Transformed message so it's ready to be accepted as input: {char_list}")
    return char_list

def extract_uniq_tokens(tokenizer):
    """Takes the first 15000 tokens that equate to a unique identifier according
    to the tokenizer, starting at index 1000 from the end of the vocabulary
    """
    uniq_token_list = []

    print("#1: Generating unique token list from model tokenizer")
    for i in tqdm(range(15000)):
        token = tokenizer.decode(len(tokenizer) - (i+1000)).replace(" ", "")
        if token.count("#") > 0:
            continue
        else:
            uniq_token_list.append(token)
    return uniq_token_list

def uniq_tokens_tensors(tokens_list, model, tokenizer, model_type):
    """Extract a dictionary of all tensors associated with tokens that contain
    the right amount of correct length floats (ref., I'm sorry, Italian
    friend!)
    """
    tensor_dict = {}
    
    print("#2: Selecting suitable tokens with correct float-sizes:")
    for token in tqdm(tokens_list):
        with torch.no_grad():
            t_id = tokenizer.convert_tokens_to_ids(token)
            run_model_type = f"model.{model_type}.embeddings.word_embeddings.weight[{t_id}]"
            t_weights = eval(run_model_type)
            target_lens = 0
            for weights in t_weights.detach().numpy():
                if len(str(weights)) >= 9 and len(str(weights)) <= 10:
                    target_lens += 1
                else:
                    continue
            if target_lens >= 25 and target_lens <= 75:
                tensor_dict[token] = t_weights
            else:
                continue
    return tensor_dict

def encrypt_text (char_list):
    """Encrypt the character list into an encrypted character list"""
    key = Fernet.generate_key()
    f = Fernet(key)
    temp_text = "".join(char_list)
    enc_text = f.encrypt(temp_text.encode('ascii'))
    enc_char_list = [char for word in enc_text.decode('ascii') for char in word]
    # print(f"After encryption, this is our character list of the original message: {enc_char_list}")
    return key.decode('ascii'), enc_char_list

def decrypt_text(key, enc_char_list):
    """Decrypt the encrypted character list"""
    key = key.encode('ascii')
    f = Fernet(key)
    temp_text = "".join(enc_char_list)
    new_enc_text = temp_text.encode('ascii')
    decrypt_text = f.decrypt(new_enc_text)
    dec_char_list = [char for word in decrypt_text.decode('ascii') for char in word]
    return dec_char_list

def add_text_tensor_dict(arr, text):
    """From the first token-tensor-pair, get the token and the accompanying
    tensor, unpack the tensor into a numpy array, and start batching in the
    character list in reverse. Go to the next token-tensort-pair if the floats
    are spent
    """
    text_batching = text
    cipher_token_dict = {}
    tensor_list = arr.items()
    # print(f"encrypted char --> original float --> transformed float: float-id
    # for cipher")
    print("#3: Adding target chars to weights in ordinal format")
    for tensor_combo in tensor_list:
        token, weights = tensor_combo
        flt_list = weights.detach().numpy()
        h_array, char_count, cipher_list = add_prog_flt(flt_list, text_batching)
        cipher_token_dict[token] = [h_array, cipher_list]
        text_batching = text_batching[:-char_count]
        if len(text_batching) == 0:
            return cipher_token_dict
        else:
            continue

def add_prog_flt(arr, prog):
    """From the numpy array and token offered, put all batched characters in the
    floats that are of a correct length, until all floats are spent. Return a
    cipher list identifying every float a character is hidden in within the
    token-set
    """
    lcount_prog = len(prog)
    lcount_arr = len(arr)
    place_chars = []
    new_arr = []
    char_count = 0

    for i in tqdm(arr):
        if lcount_prog != 0 and (len(str(i)) >= 9) and (len(str(i)) <= 10):
            str_float = str(i)
            if str_float[0] == "-":
                zeroed_str_float = str_float[:3] + str(ord(prog[lcount_prog - 1])).zfill(3) + str_float[6:]
                new_arr.append(float(zeroed_str_float))
                lcount_prog -= 1
                char_count += 1
                place_chars.append({prog[lcount_prog - 1]: (lcount_arr)})
                lcount_arr -= 1
            else:
                zeroed_str_float = str_float[:2] + str(ord(prog[lcount_prog - 1])).zfill(3) + str_float[5:]
                # print(f"{prog[lcount_prog - 1]}|{ord(prog[lcount_prog - 1])} --> {str_float} --> {zeroed_str_float}: float-id = {lcount_arr}")
                new_arr.append(float(zeroed_str_float))
                lcount_prog -= 1
                char_count += 1
                place_chars.append({prog[lcount_prog - 1]: (lcount_arr)})
                lcount_arr -= 1
        else:
            if lcount_arr != 0:
                new_arr.append(i)
                lcount_arr -= 1
        cipher_list = [char_id for char_combo in place_chars for char_id in char_combo.values()]
    return np.array(new_arr), char_count, cipher_list

def extract_text_tensor(cipher_token_list, key, tokenizer, model_type, h_model=None):
    """Extract the model based on the cipher an encryption key provided. If no
    model is given, extract the characters from the weights provided in the
    cipher token list, otherwise, extract them based on the cipher list from the
    weights in the model"""
    if h_model == None:
        text_list = []

        print(f"#4: Extracting chars in ordinal format from weights")
        for token_combo in cipher_token_list.items():
            flt_list, cipher_list = token_combo[1]
            r_payload = unpack_payload(flt_list, cipher_list)
            text_list.append("".join(r_payload))
        reversed_text_list = text_list[::-1]
        return "".join(decrypt_text(key, reversed_text_list))
    else:
        text_list = []

        print(f"#4: Extracting chars in ordinal format from weights")
        for token_combo in cipher_token_list.items():
            token = token_combo[0]
            cipher_list = token_combo[1]
            # print(token, cipher_list)
            with torch.no_grad():
                h_id = tokenizer.convert_tokens_to_ids(token)
                run_model_type = f"h_model.{model_type}.embeddings.word_embeddings.weight[{h_id}]"
                h_weights = eval(run_model_type)
                flt_list = h_weights.detach().numpy()
                r_payload = unpack_payload(flt_list, cipher_list)
                text_list.append("".join(r_payload))
        reversed_text_list = text_list[::-1]
        # print(reversed_text_list)
        return "".join(decrypt_text(key, reversed_text_list))
    
def unpack_payload(arr, cipher_list):
    """Extract all the characters in one token-cipher-list combo, given as an
    array of floats, identified by the cipher list. Put them in reverse"""
    r_payload = []
    lcount_arr = len(arr)

    for i in tqdm(arr):
        if lcount_arr in cipher_list:
            str_float = str(i)
            if str_float[0] == "-":
                r_payload.append(chr(int(str_float[3:6])))
                lcount_arr -= 1
            else:
                r_payload.append(chr(int(str_float[2:5])))
                lcount_arr -= 1
        else:
            lcount_arr -= 1
    r_payload.reverse()
    return r_payload

def poison_model(ciper_token_list, tokenizer, t_model, model_type):
    """General function to hide our message in the model, based on the total
    amount of weights needed to hide the message"""
    token_list = list(ciper_token_list.keys())
    for token in token_list:
        stego_weights = torch.tensor(ciper_token_list[token][0])
        with torch.no_grad():
            t_id = tokenizer.convert_tokens_to_ids(token)
            run_model_type = eval(f"t_model.{model_type}")
            run_model_type.embeddings.word_embeddings.weight[t_id] = stego_weights
    return t_model

def create_pure_cipher(cipher_token_list):
    """Extract only the relevant token and the float-identifiers, as we would
    only need this information to extract the characters from the correct
    weights"""
    cipher_dict = {}
    for token_combo in cipher_token_list.items():
        _, cipher = token_combo[1]
        cipher_dict[token_combo[0]] = cipher
    return cipher_dict

def save_model(t_model, tokenizer, savemodel, key, cipher):
    """Function to save the model with, induding the decryption key and the
    cipher list"""
    if savemodel != False:
        tokenizer.save_pretrained("poisoned_model")
        t_model.save_pretrained("poisoned_model")
        t_model.config.save_pretrained("poisoned_model")
        with open("poisoned_model/Key.txt", "w") as f:
            f.write(key)
        with open("poisoned_model/Cipher.json", "w") as f:
            json.dump(cipher, f)
    else:
        return "No option given to save poisoned model"
    
def load_model(model_arch, tokenizer_arch):
    """Function to load the model with, from the input-directory given, where
    the decryption key and the cipher list should be located as well"""
    loaded_model_arch = getattr(import_module('transformers'), model_arch)
    loaded_tokenizer_arch = getattr(import_module('transformers'), tokenizer_arch)
    p_model = loaded_model_arch.from_pretrained("poisoned_model")
    tokenizer = loaded_tokenizer_arch.from_pretrained("poisoned_model")
    key = ""
    cipher = {}
    with open("poisoned_model/Key.txt", "r") as f:
        key = f.read()
    with open("poisoned_model/Cipher.json", "r") as f:
        cipher = json.load(f)
    return p_model, tokenizer, key, cipher

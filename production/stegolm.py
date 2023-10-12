#       ___                       ___           ___           ___                         ___     
#      /  /\          ___        /  /\         /  /\         /  /\                       /__/\    
#     /  /:/_        /  /\      /  /:/_       /  /:/_       /  /::\                     |  |::\   
#    /  /:/ /\      /  /:/     /  /:/ /\     /  /:/ /\     /  /:/\:\    ___     ___     |  |:|:\  
#   /  /:/ /::\    /  /:/     /  /:/ /:/_   /  /:/_/::\   /  /:/  \:\  /__/\   /  /\  __|__|:|\:\ 
#  /__/:/ /:/\:\  /  /::\    /__/:/ /:/ /\ /__/:/__\/\:\ /__/:/ \__\:\ \  \:\ /  /:/ /__/::::| \:\
#  \  \:\/:/~/:/ /__/:/\:\   \  \:\/:/ /:/ \  \:\ /~~/:/ \  \:\ /  /:/  \  \:\  /:/  \  \:\~~\__\/
#   \  \::/ /:/  \__\/  \:\   \  \::/ /:/   \  \:\  /:/   \  \:\  /:/    \  \:\/:/    \  \:\      
#    \__\/ /:/        \  \:\   \  \:\/:/     \  \:\/:/     \  \:\/:/      \  \::/      \  \:\     
#      /__/:/          \__\/    \  \::/       \  \::/       \  \::/        \__\/        \  \:\    
#      \__\/                     \__\/         \__\/         \__\/                       \__\/    
#                       Made by: Hd0/Hariama | v.1.0

# This project does not encourage or allow anyone to utilize it to commit
# illegal activities. Steganography is an effective tool to counter censorship
# in countries where encryption is illegal and visibly encrypted messages may be
# incriminating. However, it can also be used to transfer malicious data. As
# such, steganography can be seen as a dual-use technology. The sole purpose of
# this project is to call attention to the "robustness"-principle of developing
# ethical 'AI'-systems. This was done purely out of educational and ethical
# interests. The vast majority of research surrounding ethical "AI"-systems in
# NLP revolves around "fairness", "accountability" and "transparancy". Beyond
# the fact that these principles deserve to be researched, I hope this project
# encourages people in academia to take the risk that aversarial ML for NLP-pipelines
# poses more serious, both as a potential threat and as a fruitful field of
# research. I truly hope more like-minded explorers join in to lay bare the
# boundaries and best-practices of MLSecOps.

# Load in functions
from local_functions import *

def main():
    """Main function that runs the different functions needed to hide and
    retrieve our message in a transformer-based language model"""
    input_dir, message, savemodel, readmodel = get_args()
    model_arch, model_type, tokenizer_arch = get_model_configs(input_dir)
    t_model, tokenizer = load_trf_modules(model_arch, tokenizer_arch, input_dir)
    if readmodel == True:
        h_model, tokenizer, key, cipher = load_model(model_arch, tokenizer_arch)
        print(f"Extracted text from poisoned model: {extract_text_tensor(cipher, key, tokenizer, model_type, h_model)}")
    if message != None:
        char_list = gen_char_list(message)
        uniq_tensor_dict = uniq_tokens_tensors(extract_uniq_tokens(tokenizer), t_model, tokenizer, model_type)
        key, enc_char_list = encrypt_text(char_list)
        cipher_token_list = add_text_tensor_dict(uniq_tensor_dict, enc_char_list)
        print(f"And after we apply the key/cipher to the model-weights, we can extract the following text: {extract_text_tensor(cipher_token_list, key, tokenizer, model_type)}")
    if savemodel == True:
        print(f"Now poisoning model of model_type {model_type}")
        poison_model(cipher_token_list, tokenizer, t_model, model_type)
        cipher = (create_pure_cipher(cipher_token_list))
        save_model(t_model, tokenizer, savemodel, key, cipher)
        print(f"Model succesfully saved!")

if __name__ == "__main__":
    main()
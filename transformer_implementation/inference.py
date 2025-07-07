import torch
import torch.nn.functional as F


def greedy_decode(model, src, max_len, start_symbol, end_symbol, device):
    model.eval()
    
    src = src.to(device)
    src_mask = create_padding_mask(src).to(device)
    
    enc_output = model.encode(src, src_mask)
    
    outputs = torch.zeros(1, 1).fill_(start_symbol).type_as(src).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = create_look_ahead_mask(outputs.size(1)).to(device)
        
        dec_output = model.decode(outputs, enc_output, src_mask, tgt_mask)
        
        pred = model.fc_out(dec_output[:, -1])
        _, next_word = torch.max(pred, dim=1)
        next_word = next_word.unsqueeze(0)
        
        outputs = torch.cat([outputs, next_word], dim=1)
        
        if next_word == end_symbol:
            break
    
    return outputs


def beam_search_decode(model, src, max_len, start_symbol, end_symbol, beam_size=4, device='cpu'):
    model.eval()
    
    src = src.to(device)
    src_mask = create_padding_mask(src).to(device)
    enc_output = model.encode(src, src_mask)
    
    outputs = torch.zeros(1, 1).fill_(start_symbol).type_as(src).to(device)
    
    fin_outputs = []
    fin_scores = []
    
    scores = torch.zeros(1).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = create_look_ahead_mask(outputs.size(1)).to(device)
        
        dec_output = model.decode(outputs, enc_output.repeat(outputs.size(0), 1, 1), 
                                src_mask.repeat(outputs.size(0), 1, 1, 1), tgt_mask)
        
        pred = model.fc_out(dec_output[:, -1])
        log_probs = F.log_softmax(pred, dim=1)
        
        scores = scores.unsqueeze(1) + log_probs
        
        if i == 0:
            scores = scores[0]
        
        scores = scores.view(-1)
        
        topk_scores, topk_indices = torch.topk(scores, beam_size)
        
        beam_indices = topk_indices // pred.size(1)
        token_indices = topk_indices % pred.size(1)
        
        next_outputs = []
        next_scores = []
        
        for j in range(beam_size):
            prev_output = outputs[beam_indices[j]]
            next_token = token_indices[j].unsqueeze(0).unsqueeze(0)
            next_output = torch.cat([prev_output, next_token], dim=1)
            
            if token_indices[j] == end_symbol:
                fin_outputs.append(next_output)
                fin_scores.append(topk_scores[j])
            else:
                next_outputs.append(next_output)
                next_scores.append(topk_scores[j])
        
        if len(next_outputs) == 0:
            break
            
        outputs = torch.cat(next_outputs, dim=0)
        scores = torch.stack(next_scores)
        
        if len(fin_outputs) >= beam_size:
            break
    
    if len(fin_outputs) == 0:
        fin_outputs = [outputs[0].unsqueeze(0)]
        fin_scores = [scores[0]]
    
    best_idx = fin_scores.index(max(fin_scores))
    return fin_outputs[best_idx]


def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, device, max_len=50):
    model.eval()
    
    tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    start_symbol = tgt_tokenizer.token_to_id('<sos>')
    end_symbol = tgt_tokenizer.token_to_id('<eos>')
    
    output = greedy_decode(model, src_tensor, max_len, start_symbol, end_symbol, device)
    
    translated = tgt_tokenizer.decode(output.squeeze(0).tolist())
    
    return translated
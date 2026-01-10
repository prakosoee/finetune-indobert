import torch

###
# Forward Function
###

# Forward function for sequence classification
# def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
#     # Unpack batch data
#     if len(batch_data) == 3:
#         (subword_batch, mask_batch, label_batch) = batch_data
#         token_type_batch = None
#     elif len(batch_data) == 4:
#         (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
#     # Prepare input & label
#     subword_batch = torch.LongTensor(subword_batch)
#     mask_batch = torch.FloatTensor(mask_batch)
#     token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
#     label_batch = torch.LongTensor(label_batch)
            
#     if device == "cuda":
#         subword_batch = subword_batch.to(device)
#         mask_batch = mask_batch.to(device)
#         token_type_batch = token_type_batch.to(device) if token_type_batch is not None else None
#         label_batch = label_batch.to(device)

#     # Forward model
#     outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
#     loss, logits = outputs[:2]
    
#     # generate prediction & label list
#     list_hyp = []
#     list_label = []
#     hyp = torch.topk(logits, 1)[1]
#     for j in range(len(hyp)):
#         list_hyp.append(i2w[hyp[j].item()])
#         list_label.append(i2w[label_batch[j][0].item()])
        
#     return loss, list_hyp, list_label
import torch

def forward_sequence_classification(
    model,
    batch_data,
    i2w=None,
    is_test=False,
    device=None,
    **kwargs
):
    # =========================
    # HANDLE BATCH DATA UNPACKING
    # =========================
    # Check if the last element is raw text (list of strings) and remove it if so.
    # This prevents errors when the user forgets to slice batch_data[:-1] in the notebook.
    if len(batch_data) > 0 and isinstance(batch_data[-1], list) and len(batch_data[-1]) > 0 and isinstance(batch_data[-1][0], str):
        batch_data = batch_data[:-1]

    if len(batch_data) == 3:
        subword_batch, mask_batch, label_batch = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        subword_batch, mask_batch, token_type_batch, label_batch = batch_data
    else:
        raise ValueError(f"Unexpected batch_data length: {len(batch_data)}. Expected 3 or 4 tensors.")

    # =========================
    # PASTIKAN SEMUA TORCH TENSOR
    # =========================
    if not torch.is_tensor(subword_batch):
        subword_batch = torch.tensor(subword_batch, dtype=torch.long)
    if not torch.is_tensor(mask_batch):
        mask_batch = torch.tensor(mask_batch, dtype=torch.long)
    if token_type_batch is not None and not torch.is_tensor(token_type_batch):
        token_type_batch = torch.tensor(token_type_batch, dtype=torch.long)
    if not torch.is_tensor(label_batch):
        label_batch = torch.tensor(label_batch, dtype=torch.long)

    # =========================
    # PINDAHKAN KE DEVICE
    # =========================
    subword_batch = subword_batch.to(device)
    mask_batch = mask_batch.to(device)
    token_type_batch = token_type_batch.to(device) if token_type_batch is not None else None
    label_batch = label_batch.to(device)

    # =========================
    # FORWARD MODEL
    # =========================
    outputs = model(
        input_ids=subword_batch,
        attention_mask=mask_batch,
        token_type_ids=token_type_batch,
        labels=label_batch
    )

    loss = outputs.loss
    logits = outputs.logits

    preds = torch.argmax(logits, dim=-1)

    return (
        loss,
        preds.detach().cpu().tolist(),
        label_batch.detach().cpu().tolist()
    )



# Forward function for word classification
def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.to(device)
        mask_batch = mask_batch.to(device)
        token_type_batch = token_type_batch.to(device) if token_type_batch is not None else None
        subword_to_word_indices_batch = subword_to_word_indices_batch.to(device)
        label_batch = label_batch.to(device)

    # Forward model
    outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]
    
    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
    for i in range(len(hyps_list)):
        hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()        
        list_hyp, list_label = [], []
        for j in range(len(hyps)):
            if labels[j] == -100:
                break
            else:
                list_hyp.append(i2w[hyps[j]])
                list_label.append(i2w[labels[j]])
        list_hyps.append(list_hyp)
        list_labels.append(list_label)
        
    return loss, list_hyps, list_labels

# Forward function for sequence multilabel classification
def forward_sequence_multi_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
            
    if device == "cuda":
        subword_batch = subword_batch.to(device)
        mask_batch = mask_batch.to(device)
        token_type_batch = token_type_batch.to(device) if token_type_batch is not None else None
        label_batch = label_batch.to(device)

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2] # logits list<tensor(bs, num_label)> ~ list of batch prediction per class 
    
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = [torch.topk(logit, 1)[1] for logit in logits] # list<tensor(bs)>
    batch_size = label_batch.shape[0]
    num_label = len(hyp)
    for i in range(batch_size):
        hyps = []
        labels = label_batch[i,:].cpu().numpy().tolist()
        for j in range(num_label):
            hyps.append(hyp[j][i].item())
        list_hyp.append([i2w[hyp] for hyp in hyps])
        list_label.append([i2w[label] for label in labels])
        
    return loss, list_hyp, list_label

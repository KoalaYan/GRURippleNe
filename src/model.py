import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


class GRURippleNet(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(GRURippleNet, self).__init__()

        self.parse_args(args, n_entity, n_relation)

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.GRU(self.dim, self.dim)
        self.criterion = nn.BCELoss()

        nn.init.xavier_normal_(self.entity_emb.weight)
        nn.init.xavier_normal_(self.relation_emb.weight)

        for name, param in self.transform_matrix.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.xavier_normal_(param)


    def parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.using_all_hops = args.using_all_hops

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self.key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings
        )
        scores = self.predict(item_embeddings, o_list)

        return_dict = self.compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict

    def compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):            
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3) # [batch_size, n_memory, dim, 1]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded)) # [batch_size, n_memory, dim]           
            v = torch.unsqueeze(item_embeddings, dim=2) # [batch_size, dim, 1]           
            probs = torch.squeeze(torch.matmul(Rh, v)) # [batch_size, n_memory]            
            probs_normalized = F.softmax(probs, dim=1) # [batch_size, n_memory]   
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2) # [batch_size, n_memory, 1]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1) # [batch_size, dim]

            if hop == 0:
                item_embeddings, h_i = self.transform_matrix(item_embeddings)
            else:
                item_embeddings, h_i = self.transform_matrix(item_embeddings, h_i)

            o_list.append(o)
        return o_list, item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import device
import math

class Network(nn.Module):
    def __init__(self, num_careers: int, num_aircrafts: int, num_origin_airports: int, 
                num_dest_airports: int, num_other_columns: int, num_hidden_layers: int, 
                activation_function: str):
        super().__init__()

        torch.set_grad_enabled(True)
        self.careers_vector_len = math.ceil(num_careers**0.25)
        self.aircrafts_vector_len = math.ceil(num_aircrafts**0.25)
        self.origin_airport_vector_len = math.ceil(num_origin_airports**0.25)
        self.dest_airport_vector_len = math.ceil(num_dest_airports**0.25)
        self.num_other_columns = num_other_columns

        self.embed_careers = nn.Embedding(num_embeddings=num_careers, embedding_dim=self.careers_vector_len).to(device)
        self.embed_aircrafts = nn.Embedding(num_embeddings=num_aircrafts, embedding_dim=self.aircrafts_vector_len).to(device)
        self.embed_origin_airports = nn.Embedding(num_embeddings=num_origin_airports, embedding_dim=self.origin_airport_vector_len).to(device)
        self.embed_dest_airports = nn.Embedding(num_embeddings=num_dest_airports, embedding_dim=self.dest_airport_vector_len).to(device)

        num_inputs = self.careers_vector_len + self.aircrafts_vector_len + self.origin_airport_vector_len + \
            self.dest_airport_vector_len + self.num_other_columns

        self.fc_list = []

        input = num_inputs
        hidden_layer_size = 1024
        key_item = 1
        for _item in range(num_hidden_layers):
            output = hidden_layer_size
            self.fc_list.append(nn.Linear(in_features=input, out_features=output).to(device))
            key = "fc" + str(key_item)
            setattr(self, key, self.fc_list[-1])
            input = output
            key_item += 1

        num_output = 1
        self.output_fc_layer = nn.Linear(in_features=input, out_features=num_output).to(device)

        self.activation_function: str = activation_function

    def _activation(self, t: torch.Tensor, activation_function: str) -> torch.Tensor:
        if activation_function.lower() == 'relu':
            return F.relu(t)
        elif activation_function.lower() == 'sigmoid':
            return torch.sigmoid(t)
        elif activation_function.lower() == 'tanh':
            return F.tanh(t)
        else:
            return t

    def forward(self, t_input: torch.Tensor) -> torch.Tensor:
        career_embeds = self.embed_careers((t_input[:,0]).long())
        aircraft_embeds = self.embed_aircrafts((t_input[:,1]).long())
        origin_airport_embeds = self.embed_origin_airports((t_input[:,2]).long())
        dest_airport_embeds = self.embed_dest_airports((t_input[:,3]).long())

        t = torch.cat((career_embeds, aircraft_embeds, origin_airport_embeds, dest_airport_embeds, t_input[:,4:]), dim=1)
        for item in self.fc_list:
            t = item(t)
            t = self._activation(t, self.activation_function)

        t = self.output_fc_layer(t)
        return t        
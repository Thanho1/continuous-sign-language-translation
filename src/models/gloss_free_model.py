import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.models.pose_encoder import PoseEncoder


class GlossFreeSLTModel(nn.Module):
    def __init__(self, gpt2_name="gpt2", unfreeze_last_n_blocks=2):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.pose_encoder = PoseEncoder(
            hidden_dim=self.gpt2.config.n_embd
        )

        # Freeze GPT-2
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # Unfreeze last N transformer blocks
        if unfreeze_last_n_blocks > 0:
            blocks = self.gpt2.transformer.h
            for block in blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

            for p in self.gpt2.transformer.ln_f.parameters():
                p.requires_grad = True

    def forward(self, pose_seq, text_input_ids, text_attention_mask):
        pose_features = self.pose_encoder(pose_seq)

        text_embeds = self.gpt2.transformer.wte(
            text_input_ids
        )

        inputs_embeds = torch.cat(
            [pose_features, text_embeds], dim=1
        )

        pose_mask = torch.ones(
            pose_features.shape[:2],
            device=pose_seq.device,
            dtype=text_attention_mask.dtype,
        )

        attention_mask = torch.cat(
            [pose_mask, text_attention_mask],
            dim=1,
        )

        ignore = torch.full(
            pose_features.shape[:2],
            -100,
            device=pose_seq.device,
            dtype=text_input_ids.dtype,
        )

        labels = torch.cat(
            [ignore, text_input_ids],
            dim=1,
        )

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pose_seq,
        max_new_tokens=40,
        num_beams=4,
    ):
        self.eval()

        pose_features = self.pose_encoder(pose_seq)

        pose_mask = torch.ones(
            pose_features.shape[:2],
            device=pose_seq.device,
            dtype=torch.long,
        )

        output = self.gpt2.generate(
            inputs_embeds=pose_features,
            attention_mask=pose_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return [
            self.tokenizer.decode(
                x,
                skip_special_tokens=True
            ).strip()
            for x in output
        ]
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

app = Flask(__name__)
CORS(app)  


class Bert_Arch(nn.Module):
    def __init__(self, bert):
        super(Bert_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

model = Bert_Arch(bert)
model.load_state_dict(torch.load("D:/python/NLP/FakeNewsDetection/pytorch_model.bin", map_location=device))
model.to(device)
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

   
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        prediction = torch.argmax(outputs, dim=1).item()

    label = "Fake News" if prediction == 1 else "Real News"


    explanation = "The text contains common patterns of misinformation." if label == "Fake News" else "The text appears factual and well-sourced."

    return jsonify({"prediction": label, "explanation": explanation})

   

if __name__ == '__main__':
    app.run(debug=True)

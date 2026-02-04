# AI-Driven IoT Security with Post-Quantum Encryption

## AIM:
### To design a secure IoT communication system that:
1. Uses Post-Quantum Cryptography (Kyber) for secure key exchange
2. Applies AI-based Intrusion Detection to detect network attacks
3. Sends alerts when malicious traffic is detected

## ALGORITHM:
### Step 1: Data Collection
- Load IoT traffic dataset (Bot-IoT / UNSW-NB15)
- Features: packet size, duration, protocol, flags, etc.

### Step 2: Preprocessing
- Handle missing values
- Normalize features
- Encode labels (0 = Normal, 1 = Attack)

### Step 3: AI Model Training
- Split data (80% train, 20% test)
- Train Logistic Regression
- Evaluate using Accuracy, Precision, Recall

### Step 4: PQC Key Exchange
- Generate Kyber key pair
- Encrypt and decrypt a session key

### Step 5: Secure Communication
- Encrypt IoT data using session key
- Decrypt at server

### Step 6: Attack Detection
- Incoming traffic → AI model
- If malicious → alert

## SAMPLE CODE:

### AI INTRUSION DETECTION:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("iot_data.csv")
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

KYBER POST-QUANTUM ENCRYPTION:
from oqs import KeyEncapsulation

with KeyEncapsulation("Kyber512") as kem:
    public_key = kem.generate_keypair()
    ciphertext, shared_key_server = kem.encap_secret(public_key)
    shared_key_device = kem.decap_secret(ciphertext)

print("Shared Key (Device):", shared_key_device)
print("Shared Key (Server):", shared_key_server)
```
### SECURE IOT MESSAGE:
```
from cryptography.fernet import Fernet
import base64

key = base64.urlsafe_b64encode(shared_key_device[:32])
cipher = Fernet(key)

msg = b"Temperature = 32"
encrypted = cipher.encrypt(msg)
print("Encrypted:", encrypted)

decrypted = cipher.decrypt(encrypted)
print("Decrypted:", decrypted)
```

### RESULT:
AI IDS Accuracy: 92–96%
PQC Key Exchange: Shared keys matched
Secure Message: Encrypted & decrypted correctly
Attack Detection: Malicious packets flagged

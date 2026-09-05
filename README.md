# Store Vision AI

Store Vision AI is a CPU-friendly retail intelligence platform combining:

- Computer Vision
- YOLO
- OpenCV
- Streamlit
- Supabase
- Inventory Analytics
- Sales Analytics
- Visitor Analytics
- Security Alerts

## Features

### Vision Scanner
Upload a store image and detect objects using YOLO.

### Inventory
Monitor:

- Stock
- Minimum stock
- Inventory value
- Low-stock products

### Sales
Monitor:

- Revenue
- Units sold
- Transactions
- Daily revenue

### Visitor Analytics
Analyze:

- Store visits
- Visitor activity
- Daily visitor counts

### Security
Monitor:

- Security alerts
- Alert severity
- Alert status

## Architecture

```text
Camera / Image / Video
          |
          v
    Vision Pipeline
     YOLO + OpenCV
          |
          v
   Retail Intelligence
          |
     +----+----+
     |         |
 Inventory   Security
     |         |
 Sales      Alerts
     |
     v
   Supabase
     |
     v
  Streamlit

curl -X POST "http://localhost:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee for orders below 149?"}'
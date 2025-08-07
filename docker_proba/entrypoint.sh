#!/bin/bash

# Ollama szerver indítása a háttérben
ollama serve &

sleep 5

# Container parancsainak futtatása
exec "$@"
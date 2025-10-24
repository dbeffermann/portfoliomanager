#!/bin/bash

echo "🔧 Activando entorno virtual..."

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    python -m venv .venv
fi

# Detectar sistema operativo (Windows vs Unix)
if [ -f ".venv/Scripts/activate" ]; then
    # Windows (Git Bash / PowerShell)
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    # Linux o macOS
    source .venv/bin/activate
else
    echo "❌ No se encontró el script de activación del entorno virtual."
    exit 1
fi

# Instalar dependencias si hay requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Instalando dependencias..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ No se encontró requirements.txt, omitiendo instalación de dependencias."
fi

echo "✅ Entorno virtual configurado y dependencias instaladas."

Instalação das dependências (Python 3.9.13) — trabalho01_U2

Passos rápidos (PowerShell, Windows):

1) Ative o venv (se já existir). No seu ambiente parece que o venv está em `.venv` — ajuste se for diferente:

```powershell
# ative o virtualenv (PowerShell)
& .\.venv\Scripts\Activate.ps1
# ou, se o caminho absoluto for preferível
& C:/Users/firer/OneDrive/Documentos/UFRN/2025.2/MLOPS/U2/.venv/Scripts/Activate.ps1
```

2) Atualize pip e instale os pacotes do `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Observação importante sobre PyTorch (CPU vs CUDA):

- As linhas no `requirements.txt` especificam versões genéricas para `torch`/`torchvision`, porém as rodas (wheels) do PyTorch dependem do CUDA e do sistema operacional.
- Para instalar a versão CPU-only (mais simples):

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

- Para instalar com suporte a CUDA (ex.: cu118) consulte a página oficial e substitua o index-url de acordo com sua GPU/CUDA:

```powershell
# exemplo para CUDA 11.8 (verifique a versão correta no https://pytorch.org/)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

Se preferir, acesse https://pytorch.org/get-started/locally/ e copie o comando recomendado (operating system, package, python, cuda).

4) (Opcional) Se for a primeira vez que usa notebooks no venv, registre o kernel:

```powershell
python -m ipykernel install --user --name trabalho01_venv --display-name "Python (trabalho01_venv)"
```

5) Execução do notebook:

- Abra o VS Code, selecione o kernel `Python (trabalho01_venv)` (ou o venv ativo) na barra inferior.
- Abra `trabalho01_U2.ipynb` e execute as células em ordem (ou `Run All`).

Notas finais:

- Se tiver problemas com instalação do PyTorch no Windows, prefira o instalador CPU para começar e depois ajuste para CUDA se precisar acelerador.
- As versões no `requirements.txt` foram escolhidas para serem compatíveis com Python 3.9.13; ajuste pins se precisar versões mais antigas/novas.

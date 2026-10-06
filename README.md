# Smart Photo Edit

Editor de imagens com IA. Você abre uma foto, descreve a edição em uma frase e o app roda um **workflow do ComfyUI** na sua própria GPU. O padrão é o **Qwen-Image 2.1** (int8) com a **LoRA Viggle Turbo** de 6 passos, e o workflow pode ser trocado em *Configurações avançadas*.

- Interface local (HTML + Tailwind + JS), servida por um backend Python (`aiohttp`)
- Funciona em **Linux, Windows e macOS** (Python 3.10+)
- Nada sai da sua máquina: o app só fala com o ComfyUI que você configurar

## Requisitos

1. **Python 3.10+**
2. **ComfyUI** instalado e funcionando, com o nó `TextEncodeQwenImage21` (testado na v0.37.4) e uma GPU NVIDIA com VRAM suficiente. O workflow padrão usa o modelo de ~7 GB mais o codificador de texto de ~9 GB, e o ComfyUI descarrega parte para a RAM; mesmo assim, **feche outros programas pesados de GPU** (servidores de LLM, por exemplo).
3. Os arquivos de modelo do Qwen-Image 2.1 (os mesmos que o template oficial do ComfyUI usa):
   `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`,
   `text_encoders/qwen3vl_8b_int8_convrot.safetensors`,
   `vae/qwen_image_2.1_vae_bf16.safetensors`

## Instalação e uso

**Linux / macOS**

```bash
./run.sh                 # cria .venv na primeira vez e abre o app em http://127.0.0.1:8765
```

**Windows**

```bat
run.bat
```

Ou, manualmente: `pip install -e .` e `python -m smart_photo_edit` (opções: `--port`, `--no-browser`, `--comfy-url`).

### Instalar o que o workflow padrão precisa (LoRA + nó customizado)

```bash
python -m smart_photo_edit setup --comfyui-dir /caminho/ComfyUI [--models-dir /caminho/models]
```

Baixa `viggle_turbo.py` para `custom_nodes/` e a LoRA `Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128` (680 MB) para `models/loras/`, com retomada se a conexão cair. **Reinicie o ComfyUI** depois. Para conferir se está tudo certo: `python -m smart_photo_edit check`.

### Conectar ao ComfyUI

Em **Configurações avançadas** (`Ctrl/⌘ + ,` ou clique na etiqueta do motor no topo):

- **Endereço**: padrão `http://127.0.0.1:8188`
- **Iniciar o ComfyUI pelo app** (opcional): informe o comando e a pasta de trabalho, por exemplo
  `C:\ComfyUI\python_embeded\python.exe main.py --port 8188` (Windows) ou
  `/caminho/ComfyUI/.venv/bin/python3 main.py --port 8188` (Linux). Se o ComfyUI estiver parado quando você clicar em **Gerar**, o app o inicia sozinho.

## Trocando o workflow

Em *Configurações avançadas → Workflow* você escolhe entre os embutidos e os que importar.

| Embutido | O que é |
| --- | --- |
| **Qwen-Image 2.1 · Viggle Turbo (6 passos)** (padrão) | Qwen-Image 2.1 int8 + LoRA Viggle Turbo v0.3, sem CFG. Rápido. |
| **Qwen-Image 2.1 · Base (25 passos)** | Só os modelos base, sem LoRA. Mais lento; usa o que você já tem instalado. |

**Importar um workflow do ComfyUI:** no ComfyUI use **Save (API Format)** e importe o `.json`. O app descobre sozinho onde ficam a imagem de entrada (nó `LoadImage`), o prompt e a semente. Ao importar, ele mostra o que faltar no seu ComfyUI (nós e arquivos de modelo) antes de você gerar.

**Workflows com parâmetros próprios** (resolução, passos, etc.) usam o formato do app, que é um envelope em volta do grafo da API:

```jsonc
{
  "format": "smart-photo-edit/workflow@1",
  "name": "Meu workflow",
  "description": "Texto curto",
  "requires": {                                  // usado por `setup` e pela verificação
    "custom_nodes": [{ "name": "x.py", "url": "https://…" }],
    "models": [{ "folder": "loras", "filename": "x.safetensors", "url": "https://…" }]
  },
  "bindings": {                                  // onde o app injeta os valores: [nó, entrada]
    "image": ["6", "image"], "prompt": ["7", "prompt"], "seed": ["9", "noise_seed"],
    "negative": ["7", "negative_prompt"],        // opcional
    "strength": ["12", "denoise"]                // opcional (+ "strength_range": [min, max])
  },
  "params": [                                    // viram controles em Configurações avançadas
    { "key": "resolution", "label": "Resolução", "type": "choice", "default": 1024,
      "options": [{ "label": "Padrão", "value": 1024 }], "bind": [["7", "resolution"]] },
    { "key": "steps", "label": "Passos", "type": "number", "default": 25, "min": 8, "max": 40, "step": 1,
      "bind": [["6", "steps"]] }
  ],
  "prompt": { /* grafo no formato API do ComfyUI */ }
}
```

Controles que o workflow não suporta (por exemplo *Intensidade* e *Evitar*, quando não há vínculo) ficam desabilitados na interface. Exemplos prontos em [`smart_photo_edit/builtin_workflows/`](smart_photo_edit/builtin_workflows). Seus workflows importados ficam em `<pasta de dados>/workflows/`.

## Atalhos

`Ctrl/⌘+K` comandos · `Ctrl/⌘+,` configurações · `Ctrl/⌘+O` abrir · `Ctrl/⌘+S` exportar · `Enter` gerar · `Esc` cancelar · `C` antes/depois (divisor) · `L` lado a lado com zoom sincronizado (`+` `−` `0`) · `Ctrl/⌘+Z` desfazer

## Dados e segurança

- Configuração, workflows importados e resultados temporários ficam em `~/.config/smart-photo-edit` (Linux), `%APPDATA%\SmartPhotoEdit` (Windows) ou `~/Library/Application Support/SmartPhotoEdit` (macOS). Defina `SPE_HOME` para mudar.
- O servidor escuta só em `127.0.0.1` e rejeita requisições com `Host`/`Origin` de outros sites. Usar `--host 0.0.0.0` expõe sua GPU à rede.
- As imagens são enviadas ao ComfyUI como arquivos **temporários** e as saídas usam `PreviewImage`, então nada se acumula na pasta `output` do ComfyUI.

## Desenvolvimento

```bash
pip install -e ".[dev]"
python -m pytest          # inclui um ComfyUI falso para testar o fluxo completo
```

A CI roda os testes em Linux, Windows e macOS. Estrutura: `smart_photo_edit/` (`server.py` rotas, `comfy.py` cliente do ComfyUI, `workflows.py` formato e validação, `service.py` execução da edição, `launcher.py`, `installer.py`, `web/index.html` interface).

## Licenças dos modelos

Este repositório não distribui pesos. A LoRA **Viggle Turbo** e o **Qwen-Image 2.1** têm termos próprios; a LoRA indica **uso não comercial** (pesquisa/avaliação). Leia a licença de cada modelo antes de usar o resultado comercialmente.

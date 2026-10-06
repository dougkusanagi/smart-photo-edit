# Smart Photo Edit

Editor de imagens com IA, com **motor local gerenciado pelo próprio app**. Você abre uma foto e descreve a edição em uma frase. O padrão é **Qwen-Image 2.1** (int8) com **LoRA Viggle Turbo** de 6 passos.

**Não precisa instalar, abrir ou configurar um ComfyUI separado.** O app prepara uma cópia privada do motor, um ambiente Python isolado e os modelos na sua pasta de dados; inicia o motor quando necessário e o encerra ao fechar o app. Configurações antigas de endereço e comando não ativam o modo externo automaticamente.

- Interface e processamento locais; interface sem CDN.
- Backend Python (`aiohttp`) em Linux, Windows e macOS.
- Depois do preparo do workflow embutido, as edições funcionam sem internet. Workflows importados podem ter dependências e comportamento próprios.

## Proposta do projeto

O Smart Photo Edit é um editor de fotografias por instruções em linguagem natural, com interface web local e motor de IA preparado e gerenciado pelo próprio aplicativo. A intenção é tornar a edição acessível em computadores pessoais, **tentando atender GPUs modestas com 6 GB de VRAM**, sem exigir uma instalação separada de ComfyUI.

Essa meta orienta as escolhas de modelos quantizados, processamento do codificador na CPU/RAM, transferência de pesos entre RAM e GPU e resoluções moderadas. Os perfis menores são configuráveis; o perfil Original ainda é o padrão e não é uma recomendação para uma GPU de 6 GB. A quantidade de VRAM, a arquitetura da placa, o driver, a RAM do sistema, a resolução e o número de referências precisam ser considerados juntos. **6 GB é uma meta de compatibilidade, não uma garantia de execução já comprovada.**

O aplicativo também mantém o prompt associado a cada edição, metadados no PNG exportado e histórico persistente de imagens. A interface deve permitir entender e revisar as instruções antes de gerar e mostrar claramente os custos e limitações das opções que exigirem mais recursos.

## Requisitos

1. **Python 3.10+** com suporte a `venv` e `pip` (3.12 ou 3.13 recomendado para os pacotes do motor).
2. **GPU compatível e memória suficiente** para o workflow. O padrão int8 é voltado a GPU NVIDIA; o suporte do servidor web em macOS não implica compatibilidade desses pesos com Metal.
3. **Internet no primeiro preparo** e espaço para cerca de **18 GB de modelos**, além do motor e suas dependências. Drivers da GPU precisam estar instalados. Em Windows/Linux, o motor instala PyTorch CUDA 13.0; use drivers compatíveis.

## Instalação e uso

**Linux / macOS**

```bash
./run.sh                 # cria .venv na primeira vez e abre o app em http://127.0.0.1:8765
```

**Windows**

```bat
run.bat
```

No PowerShell, use `./run.bat`. Copie ou extraia a pasta completa do projeto, instale Python 3.12 ou 3.13 com `pip`/`venv` e deixe o comando `py` ou `python` disponível. Não copie a `.venv` de uma instalação Linux: o script cria o ambiente do Windows na primeira execução. Também são necessários driver de GPU compatível, internet para preparar dependências/modelos e espaço em disco. O download dos modelos ocorre ao preparar o motor ou na primeira edição.

O script é o inicializador do projeto, não um executável que inclui Python e os drivers. Para testar numa GPU de 6 GB, escolha um perfil menor nas configurações antes de preparar o motor. A instalação atual usa PyTorch CUDA 13.0; GPUs NVIDIA Maxwell, Pascal e Volta não são atendidas por essa versão de CUDA. Placas dessas gerações precisam de uma combinação de dependências anterior e compatível, que o instalador ainda não seleciona automaticamente. Veja as [notas oficiais do CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/). A inferência real no Windows ainda precisa ser validada em hardware compatível.


Ou, manualmente: `pip install -e .` e `python -m smart_photo_edit` (opções: `--port`, `--no-browser`, `--comfy-url`).

### Teste na GTX 1660 Ti de 6 GB

A GTX 1660 Ti informada como equipamento de teste é da geração Turing, conforme a [NVIDIA](https://nvidianews.nvidia.com/news/new-geforce-gtx-1660-ti-delivers-great-performance-leap-for-every-gamer-starting-at-279); ela não pertence às gerações removidas do CUDA 13.0. Isso confirma o enquadramento da arquitetura, não a execução do workflow dentro de 6 GB.

Para o primeiro teste no Windows, execute `run.bat` com Python e driver compatíveis já instalados. Em **Configurações avançadas → Modelos e memória**, escolha o modelo de imagem GGUF Q4, codificador na CPU/RAM e pouca VRAM; comece em resolução 768 e uma variação. O perfil Menor usa encoder Q3 experimental, enquanto Compacto usa W4A8. A RAM disponível também precisa ser considerada. Ainda falta validar a inferência e medir o pico de VRAM nessa placa, principalmente antes de acrescentar outra referência.

### Primeiro uso

Abra uma foto, descreva a edição e clique em **Gerar**. O app baixa e prepara o motor e os modelos, mostrando o progresso. Isso pode levar vários minutos, conforme sua conexão. Para preparar antes de editar, use **Configurações avançadas → Motor local → Preparar motor local**, ou:

```bash
python -m smart_photo_edit setup
python -m smart_photo_edit check
```

O motor privado usa ComfyUI **v0.37.4**, dentro de `<pasta de dados>/engine/`, com porta local escolhida pelo app. As dependências seguem os [requisitos da versão fixa do ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/v0.37.4/requirements.txt). Os pesos base vêm do [repositório oficial do Qwen-Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Os modelos embutidos têm URLs com revisão fixa e SHA256; downloads incompletos mantêm `.part` para retomada. Ao cancelar uma edição durante o preparo, o preparo continua para o próximo uso; ao fechar o app ele é interrompido. Falhas podem ser retomadas pelo botão **Tentar preparar novamente**. O log fica em `<pasta de dados>/comfyui.log`.

### Integração externa opcional

Para desenvolvimento ou uma instalação existente, ative explicitamente com `--comfy-url` ou `SPE_COMFY_URL`. O uso normal não precisa dessa opção. O comando antigo `setup --comfyui-dir /caminho/ComfyUI` continua disponível para esse modo.

## Modelos menores e GPU de 6 GB

Em **Configurações avançadas → Modelos e memória**, cada workflow embutido salva suas próprias escolhas. Os downloads só começam ao preparar o motor ou editar uma foto.

| Perfil | Modelo de imagem Turbo | Codificador | Execução |
| --- | --- | --- | --- |
| Original | INT8, 7,26 GB + LoRA 0,68 GB | INT8, 9,35 GB | Automática |
| Compacto · CPU + GPU | GGUF Q4_K_M, 4,34 GB, Turbo incorporado | W4A8, 6,31 GB | Codificador na CPU, pouca VRAM, resolução 768 |
| Menor · Q3 experimental | GGUF Q4_K_M, 4,34 GB, Turbo incorporado | GGUF Q3_K_M, 4,12 GB + projetor visual 1,16 GB | Codificador na CPU, pouca VRAM, resolução 768 |

O encoder Q3 e seu projetor somam **5,28 GB em disco**. Isso não é uma estimativa de VRAM: pesos, ativações e operações usam memória adicional. Mesmo o perfil Menor não tem execução garantida em 6 GB. A CPU usa RAM e pode aumentar o tempo da edição. Não houve validação de inferência real desses perfis nesta implementação; os encoders GGUF são experimentais e podem perder qualidade.

Também é possível escolher Q5/Q6/Q8 no modelo de imagem e Q4 no codificador, mantendo o Qwen3-VL 8B multimodal. O app não substitui o encoder por Qwen 2.5 ou por um modelo com menos parâmetros. O VAE Qwen-Image 2.1 BF16 (0,68 GB) permanece fixo. O workflow Base tem seu próprio catálogo GGUF; o Turbo mantém seis passos, seu agendamento e a LoRA apenas quando ela não estiver incorporada aos pesos.

Fontes dos pesos: [Viggle Turbo v0.3](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo), [modelos e encoders nativos Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), [Base GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) e [Qwen3-VL 8B GGUF + projetor Unsloth](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF). O catálogo fixa revisões, tamanhos e SHA-256. O motor instala internamente o [fork ComfyUI-GGUF de leejet, revisão 373048b](https://github.com/leejet/ComfyUI-GGUF/tree/373048b8403a7820620065210a691263d4da0a61), com suporte Qwen-Image 2.1 e Qwen3-VL DeepStack, e um carregador multimodal próprio que verifica a presença do projetor e permite CPU. Nenhuma instalação separada de ComfyUI é necessária.

Trocar os modelos reinicia o motor privado na próxima preparação quando necessário. Arquivos baixados anteriormente são preservados. As opções desse catálogo não alteram workflows importados.

## Trocando o workflow

Em *Configurações avançadas → Workflow* você escolhe entre os embutidos e os que importar.

| Embutido | O que é |
| --- | --- |
| **Qwen-Image 2.1 · Viggle Turbo (6 passos)** (padrão) | Qwen-Image 2.1 int8 + LoRA Viggle Turbo v0.3, sem CFG. Rápido. |
| **Qwen-Image 2.1 · Base (25 passos)** | Só os modelos base, sem LoRA. Mais lento; o app prepara os modelos base. |

**Importar um workflow do ComfyUI:** no ComfyUI use **Save (API Format)** e importe o `.json`. O app descobre sozinho onde ficam a imagem de entrada (nó `LoadImage`), o prompt e a semente. Ao importar, ele mostra o que faltar no motor local (nós e arquivos de modelo) antes de você gerar.

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

Controles que o workflow não suporta (por exemplo *Intensidade* e *Evitar*, quando não há vínculo) ficam ocultos na interface. Exemplos prontos em [`smart_photo_edit/builtin_workflows/`](smart_photo_edit/builtin_workflows). Seus workflows importados ficam em `<pasta de dados>/workflows/`.

## Prompts e presets

O app não aplica limite de caracteres ao prompt. O motor processa o texto em tokens; a capacidade prática depende do encoder, do workflow e da memória disponível. O limite total de requisição HTTP permanece em 40 MB, incluindo a imagem.

O preset **Dourado** insere uma instrução completa e editável para integrar a pessoa com a cena: escala e perspectiva, reconstrução da iluminação, sombras de contato e projetadas, reflexos e acabamento fotográfico. A instrução fica visível antes de gerar e é salva nos metadados e no histórico como qualquer prompt. Isso não garante correção física em todo resultado; a qualidade deve ser avaliada na imagem gerada.

### Presets, limpeza e IA local

Todos os presets têm instruções detalhadas e editáveis e ficam recolhidos por padrão, acessíveis pelo botão Presets. **Limpar prompt** oferece desfazer para recuperar o texto anterior.

**Melhorar com IA** prepara automaticamente um ambiente privado e usa Qwen3-0.6B na CPU, sob demanda. A sugestão aparece para revisão e só substitui o texto quando aplicada, com opção de desfazer. É possível cancelar. O modelo trabalha apenas com texto: não analisa a fotografia. O processo termina após a sugestão, liberando sua memória antes da geração de imagem.

O primeiro uso requer internet e baixa aproximadamente 1,2 GB de pesos, além das dependências de PyTorch para CPU e Transformers. Os pesos são carregados em FP32: somente eles ocupam aproximadamente 2,4 GB de RAM, com consumo adicional durante a execução. Esse recurso não ocupa VRAM. O aprimorador aceita até 7.000 tokens de entrada; essa proteção não limita o prompt usado diretamente na edição. Os arquivos ficam em `<pasta de dados>/prompt-ai/`, e erros em `prompt-ai.log`.

### Segunda imagem de referência

Nos workflows embutidos Base e Viggle Turbo, **Adicionar referência** anexa uma segunda foto como miniatura removível acima do prompt; clicar nela insere `<image2>` no cursor. Use `<image1>` para a imagem a editar e `<image2>` para a referência. Exemplo: `Coloque na pessoa de <image1> o boné branco de <image2>, preservando sua cor, formato e detalhes.` O prompt é enviado sem instruções automáticas de fundo ou objeto. Workflows importados não recebem esse recurso automaticamente.

A referência reutiliza os mesmos modelos, mas acrescenta processamento e memória. Comece com resolução moderada e uma variação; a execução com duas fotos na GTX 1660 Ti de 6 GB ainda precisa de medição real. Os metadados guardam o prompt efetivamente enviado, o texto original e a tag, nome e hash da referência. A foto de referência não é arquivada no histórico: para reutilizá-la depois, selecione o arquivo novamente.

## Histórico e prompt de cada edição

O botão **Histórico** na barra superior abre as imagens já editadas, com data, prompt, semente e workflow. Você pode reabrir uma imagem para continuar editando, reutilizar seu prompt ou excluir a imagem do histórico. Remover uma versão da sessão ou desfazer uma edição não apaga o resultado salvo no histórico.

Cada resultado é salvo como PNG com o prompt nos campos `prompt` e `Description` e os dados da edição no campo `SmartPhotoEdit` (JSON UTF-8: prompt, semente, workflow, parâmetros e modelos). **Exportar** baixa esse PNG sem remover seus metadados. Ao abrir novamente um PNG exportado pelo app, o prompt também aparece na interface. A linha **Prompt** logo acima do campo acompanha a versão selecionada: clique nela para ler o texto inteiro ou em **Reutilizar** para levá-lo ao campo da próxima instrução.

As imagens ficam em `<pasta de dados>/results/` e permanecem após fechar ou reiniciar o app, sem expiração automática. Arquivos antigos que ainda estiverem nessa pasta aparecem no histórico; resultados anteriores sem metadados mostram “Prompt não registrado”. A cópia para a área de transferência depende do navegador e não oferece a mesma preservação de metadados que Exportar.

## Atalhos

`Ctrl/⌘+K` comandos · `Ctrl/⌘+,` configurações · `Ctrl/⌘+O` abrir · `Ctrl/⌘+S` exportar · `Enter` gerar · `Esc` cancelar, fechar painel ou voltar à visão simples · `C` antes/depois (divisor) · `L` lado a lado com zoom sincronizado (`+` `−` `0`) · `/` ir ao prompt · `Ctrl/⌘+Z` desfazer · `Ctrl/⌘+Shift+C` copiar imagem

## Dados e segurança

- Configuração, workflows importados e histórico de edições ficam em `~/.config/smart-photo-edit` (Linux), `%APPDATA%\SmartPhotoEdit` (Windows) ou `~/Library/Application Support/SmartPhotoEdit` (macOS). Defina `SPE_HOME` para mudar.
- O servidor escuta só em `127.0.0.1` e rejeita requisições com `Host`/`Origin` de outros sites. Usar `--host 0.0.0.0` expõe sua GPU à rede.
- As imagens são enviadas ao ComfyUI como arquivos **temporários** e as saídas usam `PreviewImage`, então nada se acumula na pasta `output` do ComfyUI.

## Desenvolvimento

```bash
pip install -e ".[dev]"
python -m pytest          # inclui um ComfyUI falso para testar o fluxo completo
```

O CSS da interface é distribuído pronto. Para recompilá-lo após mudar classes ou o tema (Node/npm necessários apenas no desenvolvimento):

```bash
npx --yes tailwindcss@3.4.17 -c scripts/tailwind.config.cjs -i scripts/tailwind-input.css -o smart_photo_edit/web/tailwind.css --minify
```

Os testes de integração usam um motor falso; não baixam pesos e não validam inferência em GPU real. A CI roda os testes em Linux, Windows e macOS. Estrutura: `smart_photo_edit/` (`server.py` rotas, `comfy.py` cliente do ComfyUI, `workflows.py` formato e validação, `service.py` execução da edição, `launcher.py`, `runtime.py`, `installer.py`, `web/index.html` interface).

## Licenças dos modelos

Este repositório não distribui pesos. A LoRA **Viggle Turbo** e o **Qwen-Image 2.1** têm termos próprios; a LoRA indica **uso não comercial** (pesquisa/avaliação). Leia a licença de cada modelo antes de usar o resultado comercialmente.

O tempo decorrido aparece durante a geração, incluindo preparo do motor e envio. Ao concluir, o aviso mostra o tempo total. Cada resultado guarda `duration_seconds` nos metadados e mostra sua duração na interface e no histórico: esse valor mede a execução de uma variação e a obtenção da imagem, sem o preparo inicial e o envio das referências. Resultados antigos sem duração não exibem um tempo estimado.

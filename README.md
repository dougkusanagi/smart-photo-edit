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

Para o primeiro teste no Windows, execute `run.bat` com Python e driver compatíveis já instalados. Em **Ajustes avançados → Perfil de modelos**, escolha Compacto (encoder W4A8) ou Mínimo (encoder Q3 experimental): ambos usam imagem GGUF Q4, codificador na CPU/RAM, pouca VRAM e resolução 768. Comece com uma variação. A RAM disponível também precisa ser considerada. Ainda falta validar a inferência e medir o pico de VRAM nessa placa, principalmente antes de acrescentar outra referência.

### Primeiro uso

Abra uma foto, descreva a edição e clique em **Gerar**. O app baixa e prepara o motor e os modelos, mostrando o progresso. Isso pode levar vários minutos, conforme sua conexão. Para preparar antes de editar, use **Ajustes avançados → Perfil de modelos → Baixar** (ou **Iniciar**, no topo), ou:

```bash
python -m smart_photo_edit setup
python -m smart_photo_edit check
```

O motor privado usa ComfyUI **v0.37.4**, dentro de `<pasta de dados>/engine/`, com porta local escolhida pelo app. As dependências seguem os [requisitos da versão fixa do ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/v0.37.4/requirements.txt). Os pesos base vêm do [repositório oficial do Qwen-Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Os modelos embutidos têm URLs com revisão fixa e SHA256; downloads incompletos mantêm `.part` para retomada. Ao cancelar uma edição durante o preparo, o preparo continua para o próximo uso; ao fechar o app ele é interrompido. Falhas podem ser retomadas pelo botão **Tentar preparar novamente**. O log fica em `<pasta de dados>/comfyui.log`.

### Integração externa opcional

Para desenvolvimento ou uma instalação existente, ative explicitamente com `--comfy-url` ou `SPE_COMFY_URL`. O uso normal não precisa dessa opção. O comando antigo `setup --comfyui-dir /caminho/ComfyUI` continua disponível para esse modo.

## Modelos menores e GPU de 6 GB

Em **Ajustes avançados → Perfil de modelos**, cada workflow embutido salva seu próprio perfil. Cada cartão mostra o tamanho em disco e se os arquivos já foram baixados; quando faltam arquivos, o botão **Baixar** prepara tudo antes da primeira edição, com progresso somado e cancelamento (o download retoma de onde parou). Sem clicar em Baixar, os downloads começam na primeira edição. A lista dos modelos de cada perfil fica em **Modelos usados**. O perfil Original usa resolução 1024; Compacto e Mínimo passam para 768.

| Perfil | Modelo de imagem Turbo | Codificador | Execução |
| --- | --- | --- | --- |
| Original | INT8, 7,26 GB + LoRA 0,68 GB | INT8, 9,35 GB | Automática |
| Compacto · CPU + GPU | GGUF Q4_K_M, 4,34 GB, Turbo incorporado | W4A8, 6,31 GB | Codificador na CPU, pouca VRAM, resolução 768 |
| Mínimo · Q3 experimental | GGUF Q4_K_M, 4,34 GB, Turbo incorporado | GGUF Q3_K_M, 4,12 GB + projetor visual 1,16 GB | Codificador na CPU, pouca VRAM, resolução 768 |

O encoder Q3 e seu projetor somam **5,28 GB em disco**. Isso não é uma estimativa de VRAM: pesos, ativações e operações usam memória adicional. Mesmo o perfil Mínimo não tem execução garantida em 6 GB. A CPU usa RAM e pode aumentar o tempo da edição. Não houve validação de inferência real desses perfis nesta implementação; os encoders GGUF são experimentais e podem perder qualidade.

A interface oferece só os perfis prontos. O catálogo também tem Q5/Q6/Q8 para o modelo de imagem e Q4 para o codificador, aceitos pela API (`PUT /api/settings` com `model_choices`), mantendo o Qwen3-VL 8B multimodal; uma combinação assim aparece como personalizada até que um perfil seja escolhido. As quantizações de imagem e codificador não precisam coincidir; o servidor valida as famílias do modelo, encoder, VAE, projetor visual e LoRA Turbo antes de aceitar a configuração. O app não substitui o encoder por Qwen 2.5 ou por um modelo com menos parâmetros. O VAE Qwen-Image 2.1 BF16 (0,68 GB) permanece fixo. O workflow Base tem seu próprio catálogo GGUF; o Turbo mantém seis passos, seu agendamento e a LoRA apenas quando ela não estiver incorporada aos pesos.

Fontes dos pesos: [Viggle Turbo v0.3](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo), [modelos e encoders nativos Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), [Base GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) e [Qwen3-VL 8B GGUF + projetor Unsloth](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF). O catálogo fixa revisões, tamanhos e SHA-256. O motor instala internamente o [fork ComfyUI-GGUF de leejet, revisão 373048b](https://github.com/leejet/ComfyUI-GGUF/tree/373048b8403a7820620065210a691263d4da0a61), com suporte Qwen-Image 2.1 e Qwen3-VL DeepStack, e um carregador multimodal próprio que verifica a presença do projetor e permite CPU. Nenhuma instalação separada de ComfyUI é necessária.

Trocar os modelos reinicia o motor privado na próxima preparação quando necessário. Arquivos baixados anteriormente são preservados. As opções desse catálogo não alteram workflows importados.

### Desempenho preservando a resolução

Quantização menor reduz os pesos armazenados, mas não reduz automaticamente a memória das ativações nem garante geração mais rápida. Os perfis Compacto/Mínimo começam em 768; a resolução pode ser ajustada separadamente para 1024 ou 1280 nos parâmetros do workflow, ainda sem garantia de execução em 6 GB.

- **Quantizações abaixo de Q4:** o [Base GGUF de leejet](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/main) publica Q3_K (3,27 GB) e Q2_K (2,56 GB). O fork GGUF usado pelo motor tem desquantizadores para esses formatos, mas esses pesos ainda ficam fora do catálogo do app e sua inferência não foi validada aqui. O [Turbo oficial](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) oferece Q4_K_M ou maiores; seu autor registra maior desvio de imagem em Q4. Há [conversões comunitárias Turbo Q3 (4,19 GB) e Q2 (3,77 GB)](https://huggingface.co/realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs), mas a documentação delas descreve quatro passos e diferenças de identidade/composição até antes da quantização mais agressiva; não são substituições validadas para o Turbo v0.3 de seis passos do app. Outra alternativa seria validar o Base menor com a LoRA Turbo separada, incluindo seu custo de memória e execução.
- **Encoder menor:** existe [Qwen3-VL 8B Q2_K (3,28 GB)](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/tree/main), ainda fora do catálogo. Seria necessário manter o projetor visual compatível (1,16 GB). Como os perfis menores já usam o encoder na CPU, essa troca visa principalmente RAM/download e pode afetar a compreensão da instrução e da imagem; não é uma redução direta da VRAM da difusão.
- **Cache e transferências:** o Turbo no modo de pouca VRAM já põe o cache KV na CPU sem quantizá-lo. O [nó nativo](https://github.com/Comfy-Org/ComfyUI/blob/v0.37.4/comfy_extras/nodes_qwen.py) também oferece cache INT8/INT4, que exige avaliação de qualidade antes de mudar o padrão. O [ComfyUI fixado](https://github.com/Comfy-Org/ComfyUI/blob/v0.37.4/comfy/cli_args.py) já habilita offload assíncrono por padrão na NVIDIA; `--lowvram` não tem efeito quando o gerenciamento dinâmico está ativo. O VAE nativo tenta decodificação em blocos após falta de memória, preservando as dimensões de saída, mas isso não resolve falta de memória na difusão.
- **GTX 1660 Ti:** a [seleção de precisão do ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/v0.37.4/comfy/model_management.py) trata a série 16 com cautela no FP16. Não force FP16/BF16/FP8 apenas para tentar caber na VRAM; confirme o tipo usado e a estabilidade nessa placa. Testes em uma RTX moderna não validam esse comportamento no Windows/Turing.

A LoRA adicional em `engine_nodes/spe_addon_lora.py` mantém os originais na CPU, resolve os módulos uma vez por amostragem e reutiliza uma única cópia de execução por par de pesos. Conversões são reutilizadas quando não aumentam o tamanho residente na GPU; se a execução exigir FP32 para pesos BF16, somente o par em uso é convertido temporariamente. Os hooks saem a cada chamada e as cópias de execução são liberadas ao fim da amostragem, inclusive em erro/cancelamento, antes da decodificação pelo VAE. Isso troca a retenção de pesos na GPU entre edições por uma cópia original na RAM e uma transferência em cada nova amostragem. Requer reiniciar o app/motor para carregar a versão atualizada do nó.

Verificação isolada em Linux, RTX 5060 Ti 16 GB, driver 595.91.07, PyTorch 2.14.1+cu130: os 224 pares da LoRA `Qwenimag21_c2-st2000.safetensors` ocupam 160 MiB de tensores residentes em BF16; após limpar o cache, nenhuma dessas cópias fica alocada pela LoRA na GPU (o allocator do PyTorch pode reservar a memória para reutilização). Testes numéricos em CPU e CUDA FP16/BF16/FP32 coincidiram exatamente com o cálculo anterior nos casos testados. A medição de um único ramo de atenção com 4096 tokens mostrou diferença pequena, não um ganho expressivo de geração. **Esse ensaio não executa a edição completa, não mede seu pico de VRAM e não valida 6 GB ou qualidade de imagem.** Os testes numéricos em `tests/test_addon_lora.py` precisam de PyTorch; na `.venv` do servidor sem PyTorch eles são pulados e devem ser executados também com o Python do motor privado.

## Trocando o workflow

Em *Ajustes avançados → Workflow* você escolhe entre os embutidos e os que importar.

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
  "params": [                                    // viram controles em Ajustes avançados → Geração
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

O prompt efetivo aceita até **10.000 caracteres**, incluindo o texto de apoio das LoRAs. Esse é um limite de proteção do aplicativo, não a janela de contexto do modelo. O contador inclui os complementos; ao exceder o limite, a interface bloqueia geração e aprimoramento e a API rejeita a solicitação antes do preparo do motor. O texto permanece inteiro para revisão. O aprimorador conserva também seu limite próprio de 7.000 tokens de entrada, contando as instruções internas. O limite total de requisição HTTP permanece em 40 MB, incluindo a imagem.

O preset **Dourado** insere uma instrução completa e editável para integrar a pessoa com a cena: escala e perspectiva, reconstrução da iluminação, sombras de contato e projetadas, reflexos e acabamento fotográfico. A instrução fica visível antes de gerar e é salva nos metadados e no histórico como qualquer prompt. Isso não garante correção física em todo resultado; a qualidade deve ser avaliada na imagem gerada.

### Presets, limpeza e IA local

Todos os presets têm instruções detalhadas e editáveis e ficam recolhidos por padrão, acessíveis pelo botão Presets. **Limpar prompt** oferece desfazer para recuperar o texto anterior.

**Melhorar com IA** prepara automaticamente um ambiente privado e usa Qwen3-0.6B na CPU, sob demanda. A sugestão aparece para revisão e só substitui o texto quando aplicada, com opção de desfazer. É possível cancelar. O modelo trabalha apenas com texto: não analisa a fotografia. O processo termina após a sugestão, liberando sua memória antes da geração de imagem.

O primeiro uso requer internet e baixa aproximadamente 1,2 GB de pesos, além das dependências de PyTorch para CPU e Transformers. Os pesos são carregados em FP32: somente eles ocupam aproximadamente 2,4 GB de RAM, com consumo adicional durante a execução. Esse recurso não ocupa VRAM. O aprimorador aceita até 7.000 tokens de entrada, contando também as instruções internas; esse limite é independente dos 10.000 caracteres do prompt de edição. Os arquivos ficam em `<pasta de dados>/prompt-ai/`, e erros em `prompt-ai.log`.

### Modo Remover fundo (lote)

O seletor **Editar | Remover fundo** no cabeçalho (Alt+1 / Alt+2) abre um modo só para recortar fotos, sem prompt. Adicione fotos, **escolha uma pasta** (todas as imagens dentro dela, inclusive em subpastas) ou solte arquivos e pastas na janela; cada foto vira um PNG com fundo transparente, processada uma de cada vez (até 200 por lote, 20 MB cada). Cada cartão mostra fila, progresso, erro com *tentar de novo* e, quando pronto, **Comparar** com o original e exportar. **Exportar** baixa um PNG ou, com várias prontas, um ZIP.

O PNG final conserva a resolução e os pixels de cor da foto original (com a orientação EXIF aplicada). O servidor usa apenas a transparência produzida pelo motor e preserva o perfil de cor; se necessário, redimensiona somente a máscara. A composição ocorre na CPU, sem modelo adicional. As miniaturas da lista são reduzidas; a comparação e a exportação usam a imagem completa.

O recorte usa o BiRefNet nativo do ComfyUI 0.37 (`Comfy-Org/BiRefNet`, MIT, 444 MB, revisão e SHA-256 fixos), com o mesmo grafo do template oficial "Remove Background (BiRefNet)". Ele segmenta pela forma e não pela cor, então roupas brancas sobre fundo claro não viram buracos como no preset gerativo, e não usa os modelos do Qwen: só esse arquivo é baixado. Alternar entre Editar e Remover fundo reinicia o motor privado (alguns segundos). Tempo e memória em CPU/GPU, inclusive na GTX 1660 Ti de 6 GB, **ainda não foram medidos**. O prompt registrado é o rótulo "Remover fundo (BiRefNet)".

### LoRAs adicionais (experimental)

O botão **LoRA**, ao lado de Referência e Presets, liga complementos do catálogo só na edição seguinte e funciona junto com a imagem de referência. Hoje há **Integrar luz e sombra**, que usa a LoRA [rh-qwen-image-2.1-lora](https://huggingface.co/RunningHubAI/rh-qwen-image-2.1-lora-2104918997757157378) (RunningHub, 160 MiB, revisão e SHA-256 fixos) para uniformizar a luz de objetos e produtos, realçar brilhos, criar sombra de contato e fundir o item ao cenário.

A LoRA tem um texto de apoio (gatilho `pengyu …`) que **não aparece no campo do prompt**: ele é enviado antes do seu texto. O chip acima do prompt mostra o tamanho dele em caracteres, e o contador do compositor soma os dois. Você pode gerar só com a LoRA ou acrescentar sua própria instrução. O prompt efetivamente enviado (apoio + seu texto) é salvo nos metadados e no histórico, com o seu texto em `user_prompt` e a LoRA em `addons`; ao reutilizar uma edição, o campo recebe só o que você escreveu e a LoRA volta ligada.

A LoRA é baixada na primeira vez e aplicada em tempo de execução por um nó embutido (`spe_addon_lora.py`), empilhada depois da LoRA Viggle Turbo, sem reiniciar o motor. Limites: foi treinada para objetos e produtos, não para retratos; **não foi validada** com o perfil Turbo de 6 passos, com GGUF nem em 6 GB (a combinação de duas LoRAs usa uma aproximação nos blocos MLP). A licença é do autor original; siga os termos do projeto de origem. Só funciona com o motor local gerenciado.

### Segunda imagem de referência

Nos workflows embutidos Base e Viggle Turbo, **Adicionar referência** anexa uma segunda foto como miniatura removível acima do prompt; clicar nela insere `<image2>` no cursor. Use `<image1>` para a imagem a editar e `<image2>` para a referência. Exemplo: `Coloque na pessoa de <image1> o boné branco de <image2>, preservando sua cor, formato e detalhes.` O prompt é enviado sem instruções automáticas de fundo ou objeto. Workflows importados não recebem esse recurso automaticamente.

A referência reutiliza os mesmos modelos, mas acrescenta processamento e memória. Comece com resolução moderada e uma variação; a execução com duas fotos na GTX 1660 Ti de 6 GB ainda precisa de medição real. Os metadados guardam o prompt efetivamente enviado, o texto original e a tag, nome e hash da referência. A foto de referência não é arquivada no histórico: para reutilizá-la depois, selecione o arquivo novamente.

## Histórico e prompt de cada edição

Cada item do histórico tem **Preview**, que mostra a imagem enviada e o resultado sem substituir a edição aberta. Os novos registros preservam nomes e hashes dos modelos, configuração do codificador e memória, parâmetros, semente, referência, variação e duração da execução do workflow e obtenção do resultado (sem fila, preparo inicial ou download dos modelos). Esses dados ficam também no PNG exportado. Edições antigas mostram somente os dados disponíveis; originais ou detalhes que não foram guardados não podem ser reconstruídos.

Cada resultado guarda também a **imagem original** enviada (arquivo `<id>.orig` ao lado do PNG, sem metadados). No histórico, o botão **Comparar** abre o par original → resultado com o divisor antes/depois, e **Abrir imagem** carrega os dois como versões. Isso aumenta o uso de disco em `results/`; excluir um item remove também o original. Resultados antigos não têm original.

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

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
./install.sh             # instala ou repara o ambiente do aplicativo
./run.sh                 # abre o app em http://127.0.0.1:8765
./update.sh              # com o app fechado, atualiza código e dependências
```

**Windows**

```bat
install.bat
run.bat
update.bat
```

`run` instala automaticamente se o ambiente estiver ausente ou incompleto. `install` prepara apenas o aplicativo; o motor de IA e os modelos são preparados na interface. No Linux/macOS, `SPE_PYTHON=python3.12 ./install.sh` permite escolher o Python do ambiente novo. No Windows, os scripts usam `py -3` ou `python`.

Feche o aplicativo antes de executar `update`. Em um clone Git, ele usa `git pull --ff-only` e recusa alterações locais para evitar sobrescrevê-las. Em uma instalação extraída do ZIP, baixa a versão atual de `main` no [GitHub](https://github.com/dougkusanagi/smart-photo-edit) e substitui os arquivos do aplicativo; alterações manuais nesses arquivos devem ser copiadas antes. Ambos reinstalam as dependências do servidor, preservam a `.venv` e os dados pessoais e deixam o preparo do motor/modelos a cargo do app. O modo ZIP não exige Git.

No PowerShell, use `./run.bat`. Copie ou extraia a pasta completa do projeto, instale Python 3.12 ou 3.13 com `pip`/`venv` e deixe o comando `py` ou `python` disponível. Não copie a `.venv` de uma instalação Linux: o script cria o ambiente do Windows na primeira execução. Também são necessários driver de GPU compatível, internet para preparar dependências/modelos e espaço em disco. O download dos modelos ocorre ao preparar o motor ou na primeira edição.

O script é o inicializador do projeto, não um executável que inclui Python e os drivers. Para testar numa GPU de 6 GB, escolha um perfil menor nas configurações antes de preparar o motor. A instalação atual usa PyTorch CUDA 13.0; GPUs NVIDIA Maxwell, Pascal e Volta não são atendidas por essa versão de CUDA. Placas dessas gerações precisam de uma combinação de dependências anterior e compatível, que o instalador ainda não seleciona automaticamente. Veja as [notas oficiais do CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/). A inferência real no Windows ainda precisa ser validada em hardware compatível.


Ou, manualmente: `pip install -e .` e `python -m smart_photo_edit` (opções: `--port`, `--no-browser`, `--comfy-url`).

### Gerar o ZIP para outra máquina

Execute na pasta do projeto, com Python instalado (não precisa ativar a `.venv`):

```bash
python3 scripts/build_zip.py             # Linux / macOS
```

No Windows: `py -3 scripts\build_zip.py`. O pacote sai em **`dist/smart-photo-edit.zip`**; para outro destino, use `--output /caminho/pacote.zip`. O script usa somente a biblioteca padrão, verifica a integridade e imprime o SHA-256. Inclui os arquivos atuais do app, inclusive alterações ainda não commitadas, interface compilada, workflows, catálogos, licenças, inicializadores e o próprio script. Não inclui `.venv`, Git, caches, testes, pesos, configurações pessoais ou histórico.

Na outra máquina, extraia o ZIP completo e execute `run.bat` no Windows ou `bash run.sh` no Linux/macOS, dentro da pasta `smart-photo-edit`. A instalação cria um ambiente novo e baixa as dependências e modelos necessários. Python e drivers precisam estar instalados; este ZIP é um pacote de código para instalação, não um executável nem uma instalação offline com modelos.

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

Em **Ajustes avançados → Notificar ao concluir**, você pode ativar ou desativar avisos do sistema ao terminar uma edição, upscale ou lote de remoção de fundos. A opção vem ligada; o navegador solicita permissão na primeira operação. Lotes geram um único aviso com a quantidade de fotos prontas e erros, sem avisos por foto ou ao cancelar. O app precisa permanecer aberto em uma aba; o recurso depende do suporte do navegador e das permissões do site/sistema.

## Upscale generativo

A aba **Upscale**, no topo (`Alt+3`), amplia a foto em **2× ou 4×** e permite selecionar o modelo para comparar resultados. A opção **Comparar usando o original** vem ligada: cada execução parte do mesmo original, mesmo quando outra ampliação está selecionada. Desmarque para ampliar a versão atual. A seleção do modelo fica salva neste navegador.

| Modelo | Pesos em disco | Inferência |
| --- | --- | --- |
| **SeedVR2 3B** (padrão, experimental) | 3,89 GB | Restauração em um passo na resolução escolhida, sem texto. Pesos FP8, VAE FP16 e nós nativos do ComfyUI. Saída até 8 MP. |
| **AdcSR** (experimental) | 2,16 GB | Rede SD 2.1 comprimida, sem encoder de texto nem encoder VAE. FP32, determinístico, 4× nativo; 2× reduz cada bloco. Saída até 32 MP. |
| **SinSR v1** (experimental) | 699 MB | Difusão dedicada em um passo, sem texto, 4× nativo; 2× reduz cada bloco. Saída até 32 MP. Uso não comercial. |

**SD 1.5 + ControlNet Tile e PiSA-SR foram substituídos no catálogo.** O SD 1.5 fazia 16 passos por bloco, com VAE na CPU; o PiSA-SR exigia SD 2.1 completo e encoder de texto. As opções atuais usam inferência de um passo e não recebem prompt nem controle Suave/Equilibrada/Forte. Resultados antigos continuam no histórico com seus metadados; os pesos antigos não são apagados automaticamente.

Ao selecionar uma opção, a interface mostra o estado dos pesos e o tamanho do download. **Ampliar** prepara somente o modelo escolhido. Trocar de opção reinicia o motor quando necessário, preservando os pesos baixados. Todas funcionam offline após o preparo e nenhuma precisa baixar Qwen. O ambiente privado ComfyUI/PyTorch/CUDA exige vários GB adicionais; SinSR acrescenta `timm==1.0.30` e ~34 MB de biblioteca, AdcSR acrescenta `diffusers==0.41.0`. Revisões, tamanhos e SHA-256 ficam fixados no catálogo.

O painel mostra percentual, fase e tempo decorrido. **Parar** cancela a inferência; durante o preparo inicial, o preparo continua para o próximo uso. O percentual acompanha o trabalho concluído e não prevê exatamente o tempo restante. O resultado permite comparar, desfazer, exportar PNG e reabrir pelo histórico. Desfazer mantém o histórico. Transparência, perfil de cor e original são preservados; o PNG e o histórico registram modelo, semente, escala e política de execução. Detalhes ausentes são estimados: rostos, letras e texturas podem mudar. A preferência por SeedVR2 veio de uma inspeção visual limitada de uma foto degradada, sem demonstrar superioridade universal nem recuperar detalhes verdadeiros.

A API aceita `upscale_model` (`seedvr2`, `adcsr`, `sinsr`), `upscale_seed` e `scale` (2 ou 4). O padrão é SeedVR2 e a semente padrão é 0; AdcSR é determinístico. `reconstruction` foi retirado. `GET /api/upscale/models` informa tamanhos, licenças, limites de saída e estado dos pesos. Entradas que excedem o limite de saída são recusadas antes do preparo, sem redução silenciosa. O modo exige o motor gerenciado pelo app.

**SeedVR2:** [modelo da ByteDance](https://huggingface.co/ByteDance-Seed/SeedVR2-3B), licença Apache 2.0, com [pesos convertidos pela Comfy-Org](https://huggingface.co/Comfy-Org/SeedVR2/tree/df48879708206a403d2a61acd55578c2e80fd233). O grafo usa redimensionamento Lanczos, pré-processamento SeedVR2, VAE em blocos de 512 px (sobreposição 128), um passo Euler/CFG 1 e correção de cor Lab. O ComfyUI escolhe a precisão de cálculo e gerencia offload em pouca VRAM; FP8 descreve os pesos em disco, não garante operações FP8 nativas na GTX 1660 Ti. O VAE em blocos não divide a difusão inteira: fotos grandes ainda podem esgotar a memória. O limite de 8 MP é uma proteção, não uma garantia de caber em 6 GB.

**AdcSR:** implementação adaptada da [rede do autor](https://github.com/Guaishou74851/AdcSR), com UNet SD 2.1 podado em 25% dos canais, sem texto, embedding de tempo ou encoder do VAE, e meio decodificador. Pesos Apache 2.0 e configuração da base SD 2.1 OpenRAIL++. Carregamento em dispositivo `meta` evita inicializar uma segunda cópia de GB de pesos na RAM. O checkpoint Lightning é lido com `weights_only=True`, ignorando callbacks inertes sem importar o pacote do autor. Usa FP32, blocos iniciais de 192 px da entrada, contexto e sobreposição de 32 px, redução para 128/64 px em OOM e alinhamento global de cor AdaIN. CPU e GPU seguem o dispositivo escolhido pelo motor.

**SinSR:** [arquitetura do autor](https://github.com/wyf0912/SinSR), revisão `f1735490e980162435391162ddd18c0642327c5b`, isolada em namespace próprio; licença CC BY-NC-SA 4.0. Mantém UNet FP32, ruído compartilhado na CPU, blocos de 128 px com contexto e sobreposição de 32 px. O VAE usa CUDA FP16, repete em FP32 se produzir valores não finitos e tenta blocos de 64 px antes de passar para CPU em OOM. Atenção por consultas e distâncias do codebook VQ usam fatias FP32 de até 32 MiB, preservando todos os códigos e chaves. A atenção densa anterior chegava a ~12,6 GiB de RAM; a atenção fatiada por `bmm`/softmax evita esses buffers. Não retire essa proteção com base em ensaios de imagens pequenas.

As três opções desativam o cache de nós (`--cache-none`). AdcSR e SinSR limitam CPU a quatro threads durante a execução, restauram a configuração anterior e soltam modelos e referências temporárias ao terminar, inclusive em erro/cancelamento. A montagem fica na CPU. O PNG intermediário usa compressão mínima para reduzir o custo de recompressão ao salvar com metadados.

### Medições e limites

Ensaios em **7/10/2026, Linux, RTX 5060 Ti 16 GB, driver 595.91.07, 32 GB de RAM, ComfyUI v0.37.4, PyTorch 2.14.1+cu130**, uma foto RGB, sem referência, semente 0. Esses resultados **não validam GTX 1660 Ti de 6 GB nem Windows**.

Pela API real do app, incluindo preparo local/reinício quando necessário e gravação no histórico, com pesos já baixados: AdcSR **256 × 256 → 512 × 512: 5,3 s**, **512 × 512 → 1024 × 1024: 13,3 s**; SinSR **256 × 256 → 512 × 512: 13,8 s**; SeedVR2 **256 × 256 → 512 × 512: 7,1 s** e **512 × 512 → 2048 × 2048: 19,4 s**. Os casos incluem custos diferentes de reinício e não representam um ranking rigoroso de velocidade. Todas as resoluções dos PNGs foram conferidas.

Ensaio isolado anterior do SeedVR2, allocator e memória livre reportada ao ComfyUI limitados artificialmente a **5 GiB**, pesos FP8 e VAE em blocos: entrada/saída **256/512 px: 3,2 s**, **512/1024 px: 10,0 s**, **1024/2048 px: 25,0 s**, **1500/3000 px: 53,0 s**, com picos CUDA alocados entre **3.188 e 3.967 MiB**. Entrada **2000 × 2000 → 4000 × 4000** falhou por OOM; por isso o catálogo restringe a saída a 8 MP. Tempos excluem app, download e gravação no histórico, incluem a execução do grafo; reutilização de pesos pode reduzir carregamento. Limitar o allocator não reproduz a arquitetura, o desktop ou a velocidade da GTX.

AdcSR, nó atual em **três execuções no mesmo processo**, allocator limitado a 5 GiB, entrada **512 × 512**, saída **1024 × 1024**, incluindo carregar pesos e montar os blocos: **10,01 / 8,37 / 7,87 s**. Pico CUDA **3.841 MiB alocados / 5.092 MiB reservados**, **32 MiB alocados após cada execução**; pico de RAM residente **3.441 MiB**, RAM após cada execução **1.722 / 3.076 / 3.076 MiB**. A reserva pertence ao allocator e não equivale a memória viva dos tensores. O orçamento artificial não inclui todo o consumo do driver/desktop.

SinSR, mesmo ensaio repetido com entrada **512 × 512 → 1024 × 1024**, allocator de 5 GiB: **21,18 / 20,05 / 19,91 s**, pico CUDA **1.831 MiB alocados / 3.096 MiB reservados**, **32 MiB alocados após execução**, pico RAM **3.092 MiB**, RAM final **2.544 / 2.544 / 2.952 MiB**. Esses blocos exercitam o VAE até 192 px de entrada. A estabilidade em três execuções não demonstra ausência de vazamentos em todos os fluxos.

Otimização anterior de SinSR: a atenção fatiada por matmul reduziu **192 × 192 → 2× de 10,5 para 5,3 s** e **384 × 384 → 2× de 35,0 para 14,4 s**, com saídas equivalentes às anteriores (PSNR 69 dB). São diferenças de arredondamento da implementação, não uma métrica de qualidade contra a foto original.

**GTX 1660 Ti de 6 GB / Windows: validação pendente.** O AdcSR FP32 é o primeiro candidato a testar nesse equipamento por evitar precisão reduzida e modelos auxiliares. Registre GPU, driver, versões, RAM, modelo, resolução de entrada/saída, escala, duração e pico de VRAM total/do processo antes de declarar suporte. Os testes automatizados usam motor falso e testes numéricos opcionais com PyTorch; não comprovam qualidade fotográfica ou compatibilidade de VRAM. Verificação no Chromium: catálogo/seleção dos três modelos, geração AdcSR 128 × 128 → 512 × 512, progresso, exportação PNG com metadados, comparação, reabertura pelo histórico e cancelamento. Painel conferido em desktop 1440 × 900 e celular 390 × 844, sem rolagem horizontal nem erros JavaScript. O navegador integrado T3 foi bloqueado pelo AppArmor; a verificação usou Chromium headless local. Reinicie o app para carregar o catálogo e os nós atualizados.

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

Para substituir uma pessoa pela referência, comece sem a LoRA **Integrar luz e sombra**, que foi treinada para objetos e produtos e não garante transferência de identidade. Exemplo: `Substitua completamente a pessoa de <image1> pela pessoa de <image2>, preservando a identidade, cabelo, roupas e acessórios da referência. Preserve o cenário e o enquadramento de <image1>; ajuste escala, perspectiva, iluminação e sombras de contato da nova pessoa para integrá-la à cena. Não mantenha a pessoa original.` O apoio da LoRA prioriza a edição solicitada antes da harmonização, mas a combinação continua experimental.

O painel de perfis mostra a **resolução atual**, inclusive quando ela foi alterada em Geração. Se ela diferir do padrão do perfil (1024 no Original, 768 no Compacto/Mínimo), o botão **Usar … px do perfil** restaura apenas a resolução. O encoder na CPU e duas referências em resolução alta podem levar vários minutos mesmo com seis passos; baixar a resolução reduz o trabalho, sem garantir um tempo específico ou o sucesso da edição.

Validação de um caso de substituição em 7/10/2026: Linux, RTX 5060 Ti de 16 GB, driver 595.91.07, 32 GB de RAM, ComfyUI 0.37.4 e PyTorch 2.14.1+cu130; duas imagens, resolução 768 e semente 84001916. Turbo Compacto (GGUF Q4_K_M, encoder W4A8 na CPU), sem LoRA adicional, levou 150,3 s e manteve parte da roupa original. Base Original (imagem/encoder INT8, encoder automático na GPU, 25 passos), com uma instrução que explicitava retirar mochila/saia e inserir o top, calça e pose da referência, levou 28,7 s e realizou a substituição visual, com identidade facial aproximada. Os prompts e os perfis diferem: esses tempos são medições desse caso, não um comparativo isolado dos modelos. Pico de VRAM não medido; o teste não valida execução em 6 GB. O progresso agora distingue preparação das imagens/instrução (indicando CPU quando configurada) e finalização da imagem.

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

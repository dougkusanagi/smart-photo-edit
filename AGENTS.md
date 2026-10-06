# Smart Photo Edit

## Descrição e intenção

Editor de fotografias com IA por instruções em linguagem natural. A interface web roda localmente e o backend Python/aiohttp gerencia um motor privado baseado em ComfyUI. O uso normal deve ser autocontido: o usuário não precisa instalar, abrir ou configurar um ComfyUI separado. Internet é necessária para o preparo inicial; depois do preparo, os workflows embutidos devem permitir edição local sem serviços externos.

A intenção do projeto é tentar rodar em computadores pessoais com **GPUs modestas de 6 GB de VRAM**. Trate isso como uma meta de projeto a validar, não como garantia já demonstrada. A arquitetura da GPU e os drivers também importam; 6 GB de VRAM por si só não define compatibilidade.

O equipamento informado para testar no Windows é uma **GTX 1660 Ti de 6 GB (Turing)**. Sua arquitetura não está entre as removidas do CUDA 13.0, mas a inferência deste app nessa placa ainda não foi validada. Priorize medições reais nela antes de afirmar suporte ao perfil de 6 GB.

## Funcionalidades atuais

- Workflows Qwen-Image 2.1 Base e Viggle Turbo de seis passos, com catálogo de modelos compatíveis e escolhas persistidas por workflow.
- Perfis quantizados, codificador multimodal na CPU/RAM e modo de pouca VRAM. O perfil Original continua sendo o padrão; os perfis menores e encoders GGUF não têm garantia de inferência em 6 GB.
- Prompt visível por edição, PNG exportado com metadados e histórico local persistente, com reabertura e reutilização de prompts.
- Uma imagem principal `<image1>` e uma referência opcional `<image2>` nos workflows embutidos; a inferência com duas imagens em 6 GB ainda não foi medida.
- Aprimoramento opcional com Qwen3-0.6B em FP32 na CPU, em processo privado sob demanda, com revisão, cancelamento e recuperação do texto anterior. Não analisa imagens.
- Todos os presets têm instruções detalhadas e ficam recolhidos por padrão e são acessíveis pelo botão Presets; limpar o prompt permite desfazer.

## Critérios para desenvolvimento

- Avalie o custo em VRAM, RAM, tempo, downloads e espaço em disco ao escolher dependências ou adicionar modelos. Evite manter modelos auxiliares ocupando VRAM durante a geração de imagem; prefira uso opcional, CPU e carregamento sob demanda quando viável.
- Preserve a compatibilidade entre modelo de imagem, encoder multimodal, projetor visual, VAE, LoRA e nós do workflow. Não use um encoder incompatível apenas por ser menor. Pesos Turbo com LoRA incorporada não devem receber a mesma LoRA novamente.
- Diferencie tamanho em disco de memória de execução. Só declare uma combinação compatível com 6 GB depois de testar e registrar GPU, driver, versões, RAM, modelos, resolução, número de referências e pico de VRAM.
- Ao acrescentar referências, preserve as tags `<image1>` e `<image2>` e a instrução do usuário, sem impor uma função à referência e meça o aumento de memória. Use a mesma família de modelos sempre que possível e mantenha o recurso opcional para equipamentos modestos.
- Presets devem expressar a intenção da edição com instruções editáveis. Mudanças de cenário ou luz precisam considerar identidade, proporções, escala, perspectiva, iluminação, sombras e integração, conforme o efeito solicitado.
- O aprimorador de prompt deve preservar o texto anterior, apresentar a proposta para revisão e distinguir um modelo de texto de um modelo que realmente analisa a imagem. Não descreva recursos planejados como disponíveis.
- Preserve o prompt efetivamente enviado ao workflow nos metadados, no histórico e na interface. O histórico não deve expirar automaticamente nem ser apagado ao desfazer uma edição na sessão.
- Mantenha o funcionamento em Linux e Windows. Hoje o instalador usa PyTorch CUDA 13.0, sem seleção automática de dependências para GPUs antigas; documente e valide limitações de arquitetura em vez de inferir suporte pela VRAM.

## Mapa e verificação

`smart_photo_edit/server.py`: API local; `service.py`: execução; `runtime.py` e `installer.py`: preparo e motor privado; `models.py`: catálogo e adaptação compatível dos grafos; `history.py`: resultados e metadados; `web/index.html`: interface; `builtin_workflows/`: grafos embutidos.

Use `.venv/bin/python -m pytest -q` para verificar alterações funcionais. Os testes usam um motor falso, não comprovam qualidade de imagem ou compatibilidade de VRAM. Para alterações na interface, verifique os fluxos afetados no navegador. O README explica inicialização, requisitos, modelos, dados e limitações.

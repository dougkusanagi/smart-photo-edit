"""Executado apenas no ambiente privado opcional da IA de texto."""
import json
import os
import sys


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    request = json.load(sys.stdin)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    tokenizer = AutoTokenizer.from_pretrained(request['model'], revision=request['revision'], cache_dir=request['cache'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(request['model'], revision=request['revision'], cache_dir=request['cache'], torch_dtype=torch.float32, trust_remote_code=False).to('cpu').eval()
    instruction = (
        'Você reescreve instruções para edição de fotografias. Responda somente com o prompt melhorado em português, sem explicações. '
        'Preserve todas as intenções, restrições, nomes e detalhes do texto original. Não invente elementos, não mude o objetivo. '
        'Você NÃO vê nenhuma imagem: não afirme ter analisado a foto, nem invente características da pessoa ou do cenário. '
        'Expanda pedidos vagos em 4 a 6 frases com ações concretas. Não se limite a corrigir gramática ou repetir o pedido. Torne claras as mudanças pedidas e o que deve ser preservado. Quando a edição mudar cenário ou iluminação, detalhe escala, perspectiva, '
        'iluminação coerente, sombras de contato e integração fotográfica. Para mudanças apenas de cor ou estilo, preserve composição e identidade. '
        'Preserve exatamente as tags <image1> e <image2>: <image1> identifica a imagem a editar e <image2> a referência. Nunca troque atributos como cor, material ou formato do elemento solicitado. Produza uma instrução concisa, precisa e executável, sem listas de alternativas.'
    )
    user = ('A referência está anexada como <image2>; a imagem a editar é <image1>.\n\n' if request.get('has_reference') else '') + request['prompt']
    text = tokenizer.apply_chat_template([{'role':'system', 'content':instruction}, {'role':'user', 'content':'Troque o fundo por uma praia ao pôr do sol, mantendo a pessoa.'}, {'role':'assistant', 'content':'Recrie o fundo como uma praia ao pôr do sol e integre a pessoa nessa cena. Preserve sua identidade, expressão, anatomia, pose e roupas da imagem original. Ajuste escala e perspectiva em relação ao chão e horizonte. Refaça a iluminação da pessoa e do cenário com direção de luz consistente com o sol baixo. Reconstrua sombras de contato, sombras projetadas e reflexos coerentes. Harmonize exposição, nitidez e textura para um resultado fotográfico natural.'}, {'role':'user', 'content':user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt')
    if inputs['input_ids'].shape[1] > 7000:
        raise ValueError('Este aprimorador pequeno aceita até 7.000 tokens de entrada; a edição original continua disponível sem esse limite.')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=768, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    if output[0][-1].item() != tokenizer.eos_token_id:
        raise ValueError('A sugestão excedeu o tamanho de saída da IA leve. Encurte a solicitação ou edite o texto manualmente; o original foi preservado.')
    print(json.dumps({'prompt':answer}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))

"""Workflows do ComfyUI: carregar, descobrir vínculos, aplicar valores e validar.

Um workflow do Smart Photo Edit é um JSON com este formato ("envelope"):

    {
      "format": "smart-photo-edit/workflow@1",
      "name": "Nome exibido",
      "description": "Texto curto",
      "requires": {
        "custom_nodes": [{"name": "arquivo.py", "url": "https://..."}],
        "models": [{"folder": "loras", "filename": "x.safetensors", "url": "https://...", "size_mb": 680}]
      },
      "bindings": {                        # onde o app injeta os valores
        "image":    ["6", "image"],        # obrigatório (nó LoadImage)
        "prompt":   ["7", "prompt"],       # obrigatório
        "negative": ["7", "negative_prompt"],
        "seed":     ["9", "noise_seed"],
        "strength": ["12", "denoise"]      # opcional; use "strength_range": [min, max]
      },
      "params": [                          # controles extras em "Configurações avançadas"
        {"key": "resolution", "label": "Resolução", "type": "choice",
         "options": [{"label": "Padrão", "value": 1024}], "default": 1024,
         "bind": [["7", "resolution"]]}
      ],
      "prompt": { ... grafo no formato API do ComfyUI ... }
    }

Também é aceito o JSON puro de "Save (API Format)" do ComfyUI: nesse caso os vínculos
são descobertos automaticamente (e gravados no envelope ao importar).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, field
from pathlib import Path
from typing import Any, Iterable

from . import paths

FORMAT_TAG = "smart-photo-edit/workflow@1"
Binding = tuple[str, str]  # (id do nó, nome da entrada)


class WorkflowError(ValueError):
    """Workflow inválido ou parâmetro fora do permitido (mensagem pronta para o usuário)."""


@dataclass
class Param:
    key: str
    label: str
    type: str  # "choice" | "number"
    default: Any
    bind: list[Binding]
    options: list[dict[str, Any]] = field(default_factory=list)  # choice: [{label, value}]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    hint: str = ""

    def to_public(self) -> dict[str, Any]:
        d: dict[str, Any] = {"key": self.key, "label": self.label, "type": self.type, "default": self.default, "hint": self.hint}
        if self.type == "choice":
            d["options"] = self.options
        else:
            d.update(min=self.min, max=self.max, step=self.step)
        return d

    def coerce(self, value: Any) -> Any:
        if self.type == "choice":
            for opt in self.options:
                if opt["value"] == value or str(opt["value"]) == str(value):
                    return opt["value"]
            raise WorkflowError(f"Valor inválido para “{self.label}”.")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise WorkflowError(f"“{self.label}” precisa ser um número.") from None
        if self.min is not None:
            num = max(self.min, num)
        if self.max is not None:
            num = min(self.max, num)
        if all(float(x).is_integer() for x in (self.step or 1, self.min or 0, self.default)):
            return int(round(num))
        return num


@dataclass
class Workflow:
    id: str
    name: str
    description: str
    source: str  # "builtin" | "user"
    graph: dict[str, dict[str, Any]]
    bindings: dict[str, list[Binding]]
    params: list[Param]
    requires: dict[str, list[dict[str, Any]]]
    strength_range: tuple[float, float] = (0.0, 1.0)
    path: Path | None = None
    runtime_args: list[str] = field(default_factory=list)

    @property
    def supports(self) -> dict[str, bool]:
        return {**{k: k in self.bindings for k in ("negative", "strength", "seed")},
                "reference": self.source == 'builtin' and any('images.image_1' in n['inputs'] for n in self.graph.values())}

    def to_public(self, active: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "active": active,
            "supports": self.supports,
            "params": [p.to_public() for p in self.params],
            "requires": self.requires,
        }

    def param_values(self, chosen: dict[str, Any] | None) -> dict[str, Any]:
        """Valores finais (padrão + escolhidos), já validados."""
        chosen = chosen or {}
        out: dict[str, Any] = {}
        for p in self.params:
            out[p.key] = p.coerce(chosen[p.key]) if p.key in chosen else p.default
        return out

    def apply(
        self,
        *,
        image_name: str,
        prompt: str,
        reference_name: str | None = None,
        negative: str = "",
        seed: int | None = None,
        strength: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Devolve uma cópia do grafo com os valores injetados."""
        graph = copy.deepcopy(self.graph)

        def put(targets: Iterable[Binding], value: Any) -> None:
            for node_id, name in targets:
                graph[node_id]["inputs"][name] = value

        put(self.bindings["image"], image_name)
        if reference_name:
            if not self.supports['reference']:
                raise WorkflowError('Este workflow não suporta uma segunda referência.')
            graph['_spe_reference'] = {'class_type': 'LoadImage', 'inputs': {'image': reference_name}}
            for node in graph.values():
                if 'images.image_1' in node['inputs']:
                    node['inputs']['images.image_2'] = ['_spe_reference', 0]
        put(self.bindings["prompt"], prompt)
        if "negative" in self.bindings:
            put(self.bindings["negative"], negative)
        if "seed" in self.bindings and seed is not None:
            put(self.bindings["seed"], int(seed))
        if "strength" in self.bindings and strength is not None:
            lo, hi = self.strength_range
            put(self.bindings["strength"], lo + max(0.0, min(1.0, strength)) * (hi - lo))
        values = self.param_values(params)
        for p in self.params:
            put(p.bind, values[p.key])
        return graph


# ───────────────────────── leitura ─────────────────────────

def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "workflow"


def _is_api_graph(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and bool(obj)
        and all(isinstance(v, dict) and "class_type" in v and isinstance(v.get("inputs"), dict) for v in obj.values())
    )


def _as_binding(raw: Any) -> list[Binding]:
    """Aceita ["no", "entrada"] ou [["no", "entrada"], ...]."""
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and all(isinstance(x, (str, int)) for x in raw)
    ):
        return [(str(raw[0]), str(raw[1]))]
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw):
        return [(str(a), str(b)) for a, b in raw]
    raise WorkflowError("Vínculo inválido no workflow (esperado [nó, entrada]).")


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def guess_bindings(graph: dict[str, dict[str, Any]]) -> dict[str, list[Binding]]:
    """Descobre onde ficam imagem, prompt, prompt negativo e seed num grafo API puro."""
    b: dict[str, list[Binding]] = {}
    for nid, node in graph.items():
        if node["class_type"] in ("LoadImage", "LoadImageMask") and "image" in node["inputs"]:
            b["image"] = [(nid, "image")]
            break
    text_inputs = ("prompt", "text")
    for nid, node in graph.items():
        for name in text_inputs:
            if isinstance(node["inputs"].get(name), str):
                b["prompt"] = [(nid, name)]
                break
        if "prompt" in b:
            break
    for nid, node in graph.items():
        if isinstance(node["inputs"].get("negative_prompt"), str):
            b["negative"] = [(nid, "negative_prompt")]
            break
    if "negative" not in b:  # CLIPTextEncode ligado à entrada "negative" de algum sampler
        for node in graph.values():
            link = node["inputs"].get("negative")
            if _is_link(link) and str(link[0]) in graph and isinstance(graph[str(link[0])]["inputs"].get("text"), str):
                b["negative"] = [(str(link[0]), "text")]
                break
    for nid, node in graph.items():
        for name in ("noise_seed", "seed"):
            if isinstance(node["inputs"].get(name), int):
                b["seed"] = [(nid, name)]
                break
        if "seed" in b:
            break
    return b


def _parse_params(raw: Any) -> list[Param]:
    out: list[Param] = []
    for item in raw or []:
        try:
            ptype = item["type"]
            if ptype not in ("choice", "number"):
                raise WorkflowError(f"Tipo de parâmetro desconhecido: {ptype}")
            p = Param(
                key=str(item["key"]),
                label=str(item.get("label", item["key"])),
                type=ptype,
                default=item["default"],
                bind=_as_binding(item["bind"]),
                options=[{"label": str(o["label"]), "value": o["value"]} for o in item.get("options", [])],
                min=item.get("min"),
                max=item.get("max"),
                step=item.get("step"),
                hint=str(item.get("hint", "")),
            )
        except KeyError as exc:
            raise WorkflowError(f"Parâmetro sem o campo {exc}.") from None
        if p.type == "choice" and not p.options:
            raise WorkflowError(f"O parâmetro “{p.label}” não tem opções.")
        out.append(p)
    return out


def parse_workflow(data: Any, *, wf_id: str, source: str, path: Path | None = None, fallback_name: str = "") -> Workflow:
    """Converte JSON já carregado em Workflow (valida estrutura e vínculos)."""
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        raise WorkflowError(
            "Este arquivo está no formato de interface do ComfyUI. "
            "No ComfyUI use “Save (API Format)” / “Exportar (API)” e importe esse arquivo."
        )
    meta: dict[str, Any] = {}
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict) and _is_api_graph(data["prompt"]):
        meta, graph = data, data["prompt"]
    elif _is_api_graph(data):
        graph = data
    else:
        raise WorkflowError("Não reconheci este JSON como workflow do ComfyUI em formato API.")

    graph = {str(k): v for k, v in graph.items()}
    raw_bindings = meta.get("bindings")
    bindings = {k: _as_binding(v) for k, v in raw_bindings.items()} if raw_bindings else guess_bindings(graph)
    for required, label in (("image", "imagem de entrada (nó LoadImage)"), ("prompt", "texto do prompt")):
        if required not in bindings:
            raise WorkflowError(f"Não encontrei no workflow a {label}. Defina “bindings.{required}”.")
    params = _parse_params(meta.get("params"))

    def check(targets: Iterable[Binding], what: str) -> None:
        for nid, name in targets:
            if nid not in graph:
                raise WorkflowError(f"O vínculo “{what}” aponta para um nó que não existe ({nid}).")
            if name not in graph[nid]["inputs"] and what in ("image", "prompt"):
                raise WorkflowError(f"O nó {nid} não tem a entrada “{name}” (vínculo “{what}”).")

    for k, v in bindings.items():
        check(v, k)
    for p in params:
        check(p.bind, p.key)

    rng = meta.get("strength_range") or (0.0, 1.0)
    requires = meta.get("requires") or {}
    return Workflow(
        id=wf_id,
        name=str(meta.get("name") or fallback_name or wf_id),
        description=str(meta.get("description") or ""),
        source=source,
        graph=graph,
        bindings=bindings,
        params=params,
        requires={
            "custom_nodes": list(requires.get("custom_nodes", [])),
            "models": list(requires.get("models", [])),
        },
        strength_range=(float(rng[0]), float(rng[1])),
        path=path,
    )


def load_file(path: Path, source: str) -> Workflow:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"Não consegui ler {path.name}: {exc}") from None
    return parse_workflow(data, wf_id=path.stem, source=source, path=path, fallback_name=path.stem)


def discover() -> tuple[dict[str, Workflow], dict[str, str]]:
    """Workflows embutidos + os do usuário (estes vencem em caso de mesmo id).

    Devolve (workflows, erros) onde erros mapeia nome do arquivo → mensagem.
    """
    found: dict[str, Workflow] = {}
    errors: dict[str, str] = {}
    for source, folder in (("builtin", paths.BUILTIN_WORKFLOWS_DIR), ("user", paths.user_workflows_dir())):
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.json")):
            try:
                wf = load_file(f, source)
            except WorkflowError as exc:
                errors[f.name] = str(exc)
                continue
            found[wf.id] = wf
    return found, errors


def import_text(text: str, filename: str = "") -> Workflow:
    """Importa um JSON (envelope ou API puro) para a pasta do usuário."""
    try:
        data = json.loads(text)
    except ValueError:
        raise WorkflowError("O arquivo não é um JSON válido.") from None
    stem = Path(filename).stem if filename else ""
    if isinstance(data, dict) and data.get("name"):
        base = str(data["name"])
    else:
        base = stem or "workflow importado"
    existing, _ = discover()
    wf_id = slugify(base)
    n = 2
    while wf_id in existing:  # nunca sobrescreve um workflow existente
        wf_id = f"{slugify(base)}-{n}"
        n += 1
    wf = parse_workflow(data, wf_id=wf_id, source="user", fallback_name=base)
    envelope = {
        "format": FORMAT_TAG,
        "name": wf.name,
        "description": wf.description,
        "requires": wf.requires,
        "bindings": {k: ([list(t) for t in v] if len(v) > 1 else list(v[0])) for k, v in wf.bindings.items()},
        "params": (data.get("params") if isinstance(data, dict) and "params" in data else []),
        "prompt": wf.graph,
    }
    if isinstance(data, dict) and "strength_range" in data:
        envelope["strength_range"] = data["strength_range"]
    folder = paths.user_workflows_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{wf_id}.json"
    target.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    wf.path = target
    return wf


def delete_user_workflow(wf_id: str) -> str | None:
    """Remove um workflow importado e devolve o conteúdo (para o app oferecer "Desfazer")."""
    wfs, _ = discover()
    wf = wfs.get(wf_id)
    if not wf or wf.source != "user" or not wf.path:
        return None
    text = wf.path.read_text(encoding="utf-8")
    wf.path.unlink(missing_ok=True)
    return text


# ───────────────────────── validação contra o ComfyUI ─────────────────────────

@dataclass
class Issue:
    kind: str  # "missing_node" | "missing_file"
    node_id: str
    class_type: str
    input: str = ""
    value: str = ""
    message: str = ""

    def to_public(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in ("kind", "node_id", "class_type", "input", "value", "message")}


def _combo_options(spec: Any) -> list[Any] | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    first = spec[0]
    if isinstance(first, list):
        return first
    if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options", []))
    return None


def validate(wf: Workflow, infos: dict[str, dict[str, Any]]) -> list[Issue]:
    """Confere se os nós existem no ComfyUI e se os arquivos de modelo escolhidos estão instalados.

    `infos` mapeia class_type → resposta de /object_info/<class_type> ({} se o nó não existe).
    """
    issues: list[Issue] = []
    skip = {(nid, name) for nid, name in wf.bindings.get("image", [])}
    wanted = {m.get("filename"): m for m in wf.requires.get("models", [])}
    nodes_hint = {n.get("name"): n for n in wf.requires.get("custom_nodes", [])}
    for nid, node in wf.graph.items():
        cls = node["class_type"]
        info = infos.get(cls) or {}
        if not info:
            hint = ""
            if nodes_hint:
                names = ", ".join(sorted(str(n) for n in nodes_hint))
                hint = f" Instale: {names} (rode “python -m smart_photo_edit setup”)."
            issues.append(Issue("missing_node", nid, cls, message=f"O ComfyUI não tem o nó “{cls}”.{hint}"))
            continue
        inputs_spec = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        for name, value in node["inputs"].items():
            if _is_link(value) or (nid, name) in skip or not isinstance(value, str):
                continue
            options = _combo_options(inputs_spec.get(name))
            if options is None or value in options:
                continue
            extra = ""
            if value in wanted and wanted[value].get("url"):
                extra = f" Baixe em {wanted[value]['url']}"
            issues.append(
                Issue("missing_file", nid, cls, name, value, f"Arquivo não encontrado no ComfyUI: {value}.{extra}")
            )
    return issues

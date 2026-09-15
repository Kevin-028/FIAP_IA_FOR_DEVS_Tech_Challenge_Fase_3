# Relatório de anonimização

Dados sintéticos gerados com PHI proposital (nome, CPF, telefone, e-mail, endereço)
e persistidos somente após de-identificação.

| Técnica | Aplicação |
| --- | --- |
| Pseudônimo | nome → `PAC-0001` |
| Hash com salt | CPF e número de prontuário |
| Date shifting | offset fixo por paciente (intervalos clínicos preservados) |
| Scrubbing | e-mail, telefone, CPF e endereço em texto livre |

PHI residual após scrubbing: **0**.

## Amostras antes / depois

- Nome: `Ana Beatriz Lima` → `PAC-0001`
- CPF: `390.533.447-05` → `99dc3e4dac7d`
- Prontuário: `PR-10021` → `1bebf743d8e1`
- Offset de datas: 28 dias
- Hits de PHI no texto: {'email': 1, 'phone': 1, 'cpf': 1, 'address': 1}

- Nome: `Carlos Eduardo Nunes` → `PAC-0002`
- CPF: `153.509.460-56` → `759d200254f1`
- Prontuário: `PR-10022` → `63fa38025d0d`
- Offset de datas: 39 dias
- Hits de PHI no texto: {'email': 0, 'phone': 1, 'cpf': 1, 'address': 1}

- Nome: `Fernanda Souza` → `PAC-0003`
- CPF: `231.002.999-00` → `860ade0a47f8`
- Prontuário: `PR-10023` → `4da4649038bd`
- Offset de datas: 50 dias
- Hits de PHI no texto: {'email': 1, 'phone': 0, 'cpf': 0, 'address': 1}

O conjunto bruto com PHI não é versionado. Só o SQLite anonimizado e este relatório vão ao repositório.

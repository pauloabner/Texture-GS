# Texture-GS Execution & Configuration Guide

Este repositório contém o pipeline automatizado para o treino e processamento do **Texture-GS**. O ambiente é totalmente conteinerizado utilizando Docker para garantir a consistência das dependências (CUDA 12.1, PyTorch3D, Nvdiffrast, etc.).

---

## 1. Fluxo de Execução Automática

Ao iniciar o container Docker, o script `entrypoint.sh` gerencia a execução sequencial dos seguintes passos:

1. **`train.py (gaussian3d_base.yaml)`**: Realiza o treinamento inicial do modelo base 3D Gaussian Splatting.
2. **`extract_pcd.py`**: Extrai a nuvem de pontos (`pcd.npy`) a partir do checkpoint gerado no passo 1 (iteração 30000).
3. **`train.py (uv_map_object.yaml)`**: Gera a parametrização de mapa UV para o objeto.
4. **`train.py (texture_gaussian3d.yaml)`**: Realiza o refinamento final da textura sobre os Gaussians.

---

## 2. Como Configurar os Arquivos YAML

Os arquivos de configuração `.yaml` () estão localizados na pasta `configs/`. Antes de iniciar o treino, você deve configurá-los apontando para os caminhos corretos de entrada (**datasets**) e saída (**outputs**).

Como o container utiliza volumes mapeados, você deve configurar os caminhos dentro do YAML pensando em como eles aparecem **dentro do container**:

* O diretório de dados do seu host deve ser mapeado para `/data` no container.
* O diretório de resultados/workspace do seu host deve ser mapeado para `/app/output` no container.
* O diretótio de configuração (configs/) deve ser mapeado para `/app/configs` no container.

---

### Configurações Importantes nos YAMLs

* **`data_root_dir` ou similar**: Deve apontar para pastas dentro de `/data/...` (ex: `/data/DTU/scan24`).
* **`init_from`: Deve apontar para `output/<RUN_NAME>/gaussian3d_base/checkpoints/30000.pth`. O código criará uma pasta com o nome do seu `--run_name` dentro deste diretório.
* **`pcd_load_from`: Deve apontar para `output/<RUN_NAME>/gaussian3d_base/pcd.npy`.
* **`init_uv_map_from`: Deve apontar para `output/<RUN_NAME>/uv_map/checkpoints/15000.pth`.

---

## 3. Linha de Execução do Docker

O comportamento do container foi projetado para aceitar dinamicamente o nome da execução (`--run_name`) através de uma variável de ambiente (`RUN_NAME`).

### Comando Padrão de Execução

Para rodar os três treinos em sequência utilizando a GPU H100 e definindo um nome personalizado para o treino, utilize o comando abaixo:

```bash
docker run --rm \
  -e RUN_NAME=EXPERIMENTO_01 \
  -v /mnt/data_dtu:/data \
  -v /home/user/outputs:/app/output \
  -v ./configs:/app/configs \
  --gpus all \
  --name texture-gs \
  texture-gs
```

---

## 4. Segunda Parte do Experimento: Retexturização Localizada

Após a conclusão do fluxo inicial, a segunda parte do experimento consiste em isolar uma região específica do modelo para aplicar uma nova textura.

### 4.1. Seleção da Nuvem de Pontos

O primeiro passo é obter a nuvem de pontos gerada ao final do processo anterior.

Localize o arquivo de saída: output/<RUN_NAME>/texture_gaussian3d/pcds/40000.ply.

Utilizando o software de edição de nuvens de pontos, selecione apenas a região de interesse (os pontos) que deseja retexturizar.

Salve essa seleção como um novo arquivo .ply (por exemplo, selected_region.ply) dentro do seu diretório de output para que fique acessível ao container.

### 4.2. Configurando o YAML

Abra o arquivo de configuração configs/localized_custom_gs.yaml e ajuste as variáveis necessárias. A principal configuração nesta etapa é informar ao sistema onde está a nuvem de pontos selecionada:

plyfile_gs_selected_path: Deve apontar para o caminho da nuvem de pontos selecionada no passo anterior (ex: /app/output/<RUN_NAME>/selected_region.ply).

### 4.3. Executando o Container para Retexturização

Para rodar esta segunda parte, o fluxo padrão do entrypoint.sh não é utilizado. Deve executar o container informando o script retexture.sh como argumento, garantindo o mapeamento dos mesmos volumes:

```bash
docker run --rm \
  -e RUN_NAME=EXPERIMENTO_01 \
  -v /mnt/data_dtu:/data \
  -v /home/user/outputs:/app/output \
  -v ./configs:/app/configs \
  --gpus all \
  --name texture-gs-retexture \
  texture-gs retexture.sh
```

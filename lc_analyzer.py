import numpy as np
from plyfile import PlyData, PlyElement
import os
import cv2

def analyze_texture_gs_ply(ply_path, face_resolution=512):
    """
    Analisa o arquivo PLY, filtra por lc == 0 e mapeia para a textura.
    face_resolution: a resolução de UMA face do cubemap (ex: 512x512)
    """
    if not os.path.exists(ply_path):
        print(f"Erro: Arquivo {ply_path} não encontrado.")
        return

    print(f"Carregando {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # Verificar se as propriedades necessárias existem
    available_properties = [p.name for p in vertices.properties]
    required = ['x', 'y', 'z', 'lc', 'uv_0', 'uv_1', 'uv_2', 'scale_0', 'scale_1', 'scale_2']
    
    for req in required:
        if req not in available_properties:
            print(f"Erro: Propriedade '{req}' não encontrada no arquivo PLY.")
            return

    # Extrair os dados
    x, y, z = vertices['x'], vertices['y'], vertices['z']
    lc = vertices['lc']
    uvs = np.stack([vertices['uv_0'], vertices['uv_1'], vertices['uv_2']], axis=1)

    print(f"Analisando {len(vertices)} vértices...")
    print("-" * 80)
    print(f"{'Índice':<8} | {'XYZ (Posição 3D)':<30} | {'UV (Direção)':<20} | {'Pixel XY (no Atlas)'}")
    print("-" * 80)

    count = 0
    for i in range(len(vertices)):
        if lc[i] == 0:
            count += 1
            # Coordenadas 3D
            pos_3d = (x[i], y[i], z[i])
            # Vetor UV (Direção no Cubo)
            uv_vec = uvs[i]
            
            # 1. Identificar qual face do cubo o vetor aponta (0 a 5)
            # 2. Calcular a coordenada pixel (x, y) dentro desse atlas 3x4
            face_idx, pixel_x, pixel_y = map_vector_to_atlas_pixel(uv_vec, face_resolution)

            print(f"{i:<8} | {str(np.round(pos_3d, 3)):<30} | {str(np.round(uv_vec, 3)):<20} | Face {face_idx}: ({pixel_x}, {pixel_y})")

    print("-" * 80)
    print(f"Total de pontos com lc=0 encontrados: {count}")

def modify_and_save_lc_proximal(input_path, output_path, ratio=0.2):
    """
    Seleciona um conjunto de pontos próximos (cluster) e define lc = 1.
    """
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado.")
        return

    print(f"Lendo {input_path} para modificação por proximidade...")
    plydata = PlyData.read(input_path)
    vertex_data = plydata['vertex'].data.copy()
    
    # Extrair coordenadas XYZ para cálculo de distância
    xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1)
    num_vertices = len(vertex_data)
    num_to_change = int(num_vertices * ratio)

    # 1. Escolher um ponto aleatório como "semente"
    seed_idx = np.random.randint(0, num_vertices)
    seed_pos = xyz[seed_idx]

    # 2. Calcular distâncias de todos os pontos em relação à semente e pegar os N mais próximos
    distances = np.linalg.norm(xyz - seed_pos, axis=1)
    indices = np.argpartition(distances, num_to_change)[:num_to_change]

    vertex_data['lc'][indices] = 1.0

    print(f"Salvando cluster de {num_to_change} pontos com lc=1 em: {output_path}")
    PlyData([PlyElement.describe(vertex_data, 'vertex')], text=False).write(output_path)

def create_chromakey_texture(ply_path, input_texture_path, output_texture_path):
    """
    Gera uma nova textura onde os pixels correspondentes a lc=1 assumem a cor verde limão.
    Os pixels com lc=0 mantêm a cor original da textura de entrada.
    """
    if not os.path.exists(ply_path):
        print(f"Erro: Arquivo {ply_path} não encontrado.")
        return
    if not os.path.exists(input_texture_path):
        print(f"Erro: Textura original {input_texture_path} não encontrada.")
        return

    print(f"Carregando dados para gerar textura Chroma Key...")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # Carregar a imagem original (OpenCV usa BGR por padrão)
    img = cv2.imread(input_texture_path)
    h, w, _ = img.shape
    
    # A resolução da face do cubemap no atlas 3x4 (face_res = altura / 3)
    res = h // 3

    lc = vertices['lc']
    uvs = np.stack([vertices['uv_0'], vertices['uv_1'], vertices['uv_2']], axis=1)
    # Carregar as escalas (elas estão em log-space no PLY original do 3DGS)
    scales = np.stack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']], axis=1)

    print(f"Processando {len(vertices)} Gaussianos...")
    count = 0
    for i in range(len(vertices)):
        if lc[i] == 1:
            # Mapear vetor UV (direção) para coordenadas XY no atlas
            face_idx, px, py = map_vector_to_atlas_pixel(uvs[i], res)
            
            # Calcular o raio baseado na escala física da Gaussiana
            # Convertemos do log-space e pegamos a maior dimensão
            max_scale = np.exp(scales[i]).max()
            # Heurística: escala * resolução_da_face * multiplicador (ex: 2.0 para garantir cobertura)
            radius = int(max_scale * res * 2.0)
            radius = max(1, radius) # Garante pelo menos 1 pixel

            if 0 <= py < h and 0 <= px < w:
                cv2.circle(img, (px, py), radius, (0, 255, 0), -1)
                count += 1

    # Criar uma máscara para manter apenas as áreas válidas do cubemap (formato de cruz 3x4)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:res, res:2*res] = 255       # Face Top
    mask[res:2*res, 0:w] = 255         # Fileira do meio (4 faces laterais)
    mask[2*res:3*res, res:2*res] = 255 # Face Bottom
    img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(output_texture_path, img)
    print(f"Nova textura salva com sucesso em: {output_texture_path}")
    print(f"Total de pixels alterados para Chroma Key: {count}")

def create_mask_texture(ply_path, input_texture_path, output_mask_path):
    """
    Gera uma nova imagem de máscara (preto e branco).
    Os pixels correspondentes a lc=1 assumem a cor branca.
    Os demais pixels ficam pretos.
    """
    if not os.path.exists(ply_path):
        print(f"Erro: Arquivo {ply_path} não encontrado.")
        return
    if not os.path.exists(input_texture_path):
        print(f"Erro: Textura original {input_texture_path} não encontrada para referência de tamanho.")
        return

    print(f"Carregando dados para gerar máscara binária...")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # Usamos a imagem original apenas para obter as dimensões h e w
    ref_img = cv2.imread(input_texture_path)
    h, w, _ = ref_img.shape
    res = h // 3

    # Iniciar com uma imagem totalmente preta
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)

    lc = vertices['lc']
    uvs = np.stack([vertices['uv_0'], vertices['uv_1'], vertices['uv_2']], axis=1)
    scales = np.stack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']], axis=1)

    for i in range(len(vertices)):
        if lc[i] == 1:
            face_idx, px, py = map_vector_to_atlas_pixel(uvs[i], res)
            max_scale = np.exp(scales[i]).max()
            radius = int(max_scale * res * 2.0)
            radius = max(1, radius)

            if 0 <= py < h and 0 <= px < w:
                # Onde for lc=1, pintamos de branco (255, 255, 255)
                cv2.circle(mask_img, (px, py), radius, (255, 255, 255), -1)

    # Garante que apenas a área da cruz do cubemap seja considerada (limpa sangramentos nas bordas)
    layout_mask = np.zeros((h, w), dtype=np.uint8)
    layout_mask[0:res, res:2*res] = 255
    layout_mask[res:2*res, 0:w] = 255
    layout_mask[2*res:3*res, res:2*res] = 255
    mask_img = cv2.bitwise_and(mask_img, mask_img, mask=layout_mask)

    cv2.imwrite(output_mask_path, mask_img)
    print(f"Máscara binária salva com sucesso em: {output_mask_path}")

def map_vector_to_atlas_pixel(v, res):
    """
    Converte um vetor direção 3D em coordenadas de pixel em um atlas 3x4.
    Baseado no layout do cube_map() do Texture-GS.
    """
    abs_v = np.abs(v)
    mag = np.max(abs_v)
    v /= (mag + 1e-8) # Normaliza pelo eixo maior
    
    # Determinar a face (0:Right, 1:Left, 2:Top, 3:Bottom, 4:Front, 5:Back)
    if abs_v[0] == mag: # Eixo X
        face = 0 if v[0] > 0 else 1
        u, v_coord = (-v[2], -v[1]) if v[0] > 0 else (v[2], -v[1])
    elif abs_v[1] == mag: # Eixo Y
        face = 2 if v[1] > 0 else 3
        u, v_coord = (v[0], v[2]) if v[1] > 0 else (v[0], -v[2])
    else: # Eixo Z
        face = 4 if v[2] > 0 else 5
        u, v_coord = (v[0], -v[1]) if v[2] > 0 else (-v[0], -v[1])

    # Converter u, v de [-1, 1] para [0, res-1]
    lx = int(((u + 1) / 2) * (res - 1))
    ly = int(((v_coord + 1) / 2) * (res - 1))

    # Mapear para a posição global no atlas 3x4 (conforme implementado no TextureGaussian3D.cube_map)
    # Layout Atlas:
    # [  , F2,  ,  ]  (Top)
    # [F1, F4, F0, F5] (Left, Front, Right, Back)
    # [  , F3,  ,  ]  (Bottom)
    
    face_offsets = {
        2: (res, 0),        # Top (F2)
        1: (0, res),        # Left (F1)
        4: (res, res),      # Front (F4)
        0: (2*res, res),    # Right (F0)
        5: (3*res, res),    # Back (F5)
        3: (res, 2*res)     # Bottom (F3)
    }
    
    offset_x, offset_y = face_offsets[face]
    return face, offset_x + lx, offset_y + ly

def apply_external_texture_by_mask(ply_path, input_texture_path, external_texture_path, output_path):
    """
    Aplica uma textura externa sobre a textura original usando a lógica de lc=1 do PLY.
    Redimensiona a textura externa se as dimensões forem diferentes da máscara.
    """
    if not all(os.path.exists(p) for p in [ply_path, input_texture_path, external_texture_path]):
        print("Erro: Um ou mais arquivos de entrada não foram encontrados.")
        return

    # 1. Carregar texturas
    original_img = cv2.imread(input_texture_path)
    external_img = cv2.imread(external_texture_path)
    h, w, _ = original_img.shape

    # 2. Redimensionar textura externa se necessário
    if external_img.shape[0] != h or external_img.shape[1] != w:
        print(f"Redimensionando textura externa de {external_img.shape[:2]} para {(h, w)}...")
        external_img = cv2.resize(external_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Gerar a máscara baseada no PLY (lc=1)
    print("Gerando máscara de aplicação...")
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    res = h // 3
    
    # Máscara binária (começa preta)
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    lc = vertices['lc']
    uvs = np.stack([vertices['uv_0'], vertices['uv_1'], vertices['uv_2']], axis=1)
    scales = np.stack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']], axis=1)

    for i in range(len(vertices)):
        if lc[i] == 1:
            _, px, py = map_vector_to_atlas_pixel(uvs[i], res)
            max_scale = np.exp(scales[i]).max()
            radius = int(max_scale * res * 2.0)
            radius = max(1, radius)

            if 0 <= py < h and 0 <= px < w:
                cv2.circle(mask_img, (px, py), radius, (255, 255, 255), -1)

    # Limpeza do layout do cubemap
    layout_mask = np.zeros((h, w), dtype=np.uint8)
    layout_mask[0:res, res:2*res] = 255
    layout_mask[res:2*res, 0:w] = 255
    layout_mask[2*res:3*res, res:2*res] = 255
    mask_img = cv2.bitwise_and(mask_img, mask_img, mask=layout_mask)

    # 4. Operação de "Convolução"/Blending
    # Onde a máscara é branca (255), usamos a textura externa.
    # Onde é preta (0), mantemos a original.
    mask_bool = mask_img == 255
    result_img = np.where(mask_bool, external_img, original_img)

    # Salvar resultado
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    cv2.imwrite(output_path, result_img)
    print(f"Textura combinada salva com sucesso em: {output_path}")

if __name__ == "__main__":
    # Altere para o caminho do seu arquivo salvo
    path_to_ply = "output/texture_gaussian3d/2026-04-24_15-50-35/pcds/15000.ply"
    path_modified_ply = "output/texture_gaussian3d/modified.ply"
    
    # Caminhos para as texturas
    path_input_texture = "output/texture.png"  # Sua textura original
    path_output_texture = "output/texture_chromakey.png"

    # 1. Modifica 20% dos pontos (agrupados por proximidade) para lc=1 e salva
    modify_and_save_lc_proximal(path_to_ply, path_modified_ply, ratio=0.01)

    # 2. Cria a nova textura com efeito Chroma Key nos pontos lc=1
    create_chromakey_texture(path_modified_ply, path_input_texture, path_output_texture)

    # 3. Gera a máscara P&B
    create_mask_texture(path_modified_ply, path_input_texture, "output/mask.png")

    # 4. Aplica textura externa baseada na máscara
    path_external_tex = "assets/textures/white-marble-texture-close-up.jpg"
    path_combined_out = "output/combined_texture.png"
    apply_external_texture_by_mask(path_modified_ply, path_input_texture, path_external_tex, path_combined_out)

    # 5. Analisar o arquivo modificado (opcional)
    #analyze_texture_gs_ply(path_modified_ply, face_resolution=1024)

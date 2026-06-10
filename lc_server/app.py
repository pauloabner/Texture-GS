import os
import subprocess
import shutil
import yaml
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Por padrão, o Flask procura templates na pasta './templates'.
# Como o index.html está na mesma pasta que este arquivo, 
# configuramos o template_folder para a pasta atual ('.').
app = Flask(__name__, template_folder='.')

# Configuration for upload and log folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LOG_FOLDER'] = LOG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size
app.config['HOST_DATA_DIR'] = '/mnt/abner/data_dtu'

ALLOWED_EXTENSIONS = {'yaml'}

def _run_docker_container(docker_command, log_filename, description):
    """Função auxiliar para executar o Docker e gerenciar logs."""
    log_filepath = os.path.join(app.config['LOG_FOLDER'], log_filename)
    try:
        with open(log_filepath, 'w') as log_file:
            process = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                check=False
            )
            log_file.write(f"--- {description} ---\n")
            log_file.write(" ".join(docker_command) + "\n\n")
            log_file.write("--- Standard Output ---\n")
            log_file.write(process.stdout)
            log_file.write("\n--- Standard Error ---\n")
            log_file.write(process.stderr)
            log_file.write(f"\n--- Exit Code: {process.returncode} ---\n")
        return True
    except Exception as e:
        if os.path.exists(log_filepath):
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"\n--- Error during execution: {e} ---\n")
        return False

def get_datasets_list():
    """Lista as pastas dentro do diretório de dados para o combo box."""
    data_path = app.config['HOST_DATA_DIR']
    try:
        return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    except Exception:
        return []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_logs_list():
    """Retorna uma lista de arquivos de log ordenados pelo mais recente."""
    if not os.path.exists(app.config['LOG_FOLDER']):
        return []
    files = [f for f in os.listdir(app.config['LOG_FOLDER']) if f.endswith('.txt')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['LOG_FOLDER'], x)), reverse=True)

    logs_data = []
    for filename in files:
        try:
            name_without_ext = filename[:-4]
            # Trata logs normais e logs de RETEXTURE
            if name_without_ext.startswith('RETEXTURE_'):
                run_name = name_without_ext.replace('RETEXTURE_', '')
            else:
                run_name = name_without_ext

            if run_name:
                # Novo Caminho consolidado: uploads/outputs/<run_name>/texture_gaussian3d/pcds/40000.ply
                ply_path = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name, 'texture_gaussian3d', 'pcds', '40000.ply')
                exists = os.path.exists(ply_path)

                # Verifica se existe o arquivo de retexturização combinada
                combined_ply_path = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name, 'localized_custom_gs', 'combined_texture_ply.ply')
                combined_exists = os.path.exists(combined_ply_path)

                logs_data.append({
                    'filename': filename,
                    'run_name': run_name,
                    'ply_exists': exists,
                    'combined_ply_exists': combined_exists,
                })
            else:
                logs_data.append({'filename': filename, 'run_name': None, 'ply_exists': False, 'combined_ply_exists': False})
        except Exception:
            logs_data.append({'filename': filename, 'run_name': None, 'ply_exists': False, 'combined_ply_exists': False})
    
    return logs_data

@app.route('/')
def index():
    return render_template('index.html', logs=get_logs_list())

@app.route('/datasets')
def datasets_page():
    return render_template('datasets.html', datasets=get_datasets_list(), logs=get_logs_list())

@app.route('/start_from_dataset', methods=['POST'])
def start_from_dataset():
    run_name = secure_filename(request.form.get('run_name', 'DEFAULT_EXPERIMENT'))
    dataset_path = request.form.get('dataset_path')

    if not dataset_path:
        return "Nenhum dataset selecionado", 400

    # 1. Definir caminhos baseados na estrutura uploads/outputs/<RUN_NAME>
    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name)
    os.makedirs(target_dir, exist_ok=True)

    # 2. Copiar arquivos da pasta 'template' para dentro da pasta do RUN_NAME
    template_path = os.path.join(os.path.dirname(__file__), 'templates')
    if os.path.exists(template_path):
        shutil.copytree(template_path, target_dir, dirs_exist_ok=True)

        # 3. Substituir placeholders nos arquivos YAML copiados
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.yaml'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Realiza a substituição dos valores informados
                    new_content = content.replace('<DTU_SELECTED>', dataset_path).replace('<RUN_NAME>', run_name)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

        # 4. Execução do Docker (Idêntico ao processo do index)
        abs_target_dir = os.path.abspath(target_dir) # Onde os YAMLs foram copiados
        abs_host_output_dir = os.path.dirname(abs_target_dir) # Pai da pasta do experimento (uploads/outputs)

        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f"{app.config['HOST_DATA_DIR']}:/data",
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_target_dir}:/app/configs', # Monta a pasta com os YAMLs processados
            '--gpus', 'all',
            '--name', f'texture-gs-{run_name}',
            'texture-gs'
        ]

        log_filename = f'{run_name}.txt'
        success = _run_docker_container(docker_command, log_filename, "Docker Command (from Dataset Page)")

        if success:
            return render_template('datasets.html', log_link=url_for('get_log', filename=log_filename), logs=get_logs_list(), datasets=get_datasets_list())
        else:
            return f"Erro ao executar Docker. Veja o log: <a href='{url_for('get_log', filename=log_filename)}'>{log_filename}</a>", 500

    return "Erro: Pasta de templates não encontrada", 500

@app.route('/run_texture_gs', methods=['POST'])
def run_texture_gs():
    if request.method == 'POST':
        run_name = secure_filename(request.form.get('run_name', 'DEFAULT_EXPERIMENT'))
        
        # Estrutura consolidada: uploads/outputs/<run_name>
        run_config_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name)
        os.makedirs(run_config_dir, exist_ok=True)

        config_files_uploaded = []
        config_file_names = [
            'gaussian3d_base.yaml',
            'uv_map.yaml',
            'texture_gaussian3d.yaml'
        ]

        for field_name in config_file_names:
            if field_name in request.files:
                file = request.files[field_name]
                if file.filename == '':
                    return f"No selected file for {field_name}", 400
                if file and allowed_file(file.filename):
                    filename = secure_filename(field_name) # Use predefined name for consistency
                    file_path = os.path.join(run_config_dir, filename)
                    file.save(file_path)
                    print(f"[*] Configuração salva em: {file_path}")
                    config_files_uploaded.append(file_path)
                else:
                    return f"Invalid file type for {field_name}. Only .yaml files are allowed.", 400
            else:
                return f"Missing config file: {field_name}", 400

        # Define host paths for data and output, these should match your Docker setup
        # IMPORTANT: Adjust these paths to your actual host system's data and output directories
        
        # Get the absolute path of the run_config_dir for Docker volume mounting
        abs_run_config_dir = os.path.abspath(run_config_dir)
        abs_host_output_dir = os.path.dirname(abs_run_config_dir) # Pai da pasta do experimento

        # Construct the Docker command
        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f"{app.config['HOST_DATA_DIR']}:/data",
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_run_config_dir}:/app/configs', # Mount the uploaded configs
            '--gpus', 'all',
            '--name', f'texture-gs-{run_name}',
            'texture-gs'
        ]

        log_filename = f'{run_name}.txt'
        success = _run_docker_container(docker_command, log_filename, "Docker Command (from Upload Page)")

        if success:
            return render_template('index.html', log_link=url_for('get_log', filename=log_filename), logs=get_logs_list())
        else:
            return f"Erro ao executar Docker. Veja o log: <a href='{url_for('get_log', filename=log_filename)}'>{log_filename}</a>", 500

@app.route('/logs/<filename>')
def get_log(filename):
    return send_from_directory(app.config['LOG_FOLDER'], filename)

@app.route('/download_ply/<run_name>')
def download_ply(run_name):
    # Reconstrói o caminho para o arquivo 40000.ply
    directory = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name, 'texture_gaussian3d', 'pcds')
    filename = '40000.ply'
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/download_combined_ply/<run_name>')
def download_combined_ply(run_name):
    # Reconstrói o caminho para o arquivo combined_texture_ply.ply gerado na retexturização
    directory = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name, 'localized_custom_gs')
    filename = 'combined_texture_ply.ply'
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/delete_run/<filename>')
def delete_run(filename):
    # 1. Remove o arquivo de log
    log_path = os.path.join(app.config['LOG_FOLDER'], filename)
    
    # Extrai o run_name do log (removendo RETEXTURE_ se houver e a extensão .txt)
    run_name = filename[:-4]
    if run_name.startswith('RETEXTURE_'):
        run_name = run_name.replace('RETEXTURE_', '')

    if os.path.exists(log_path):
        os.remove(log_path)
    
    # 2. Remove a pasta de processamento (uploads/<run_name>)
    if run_name:
        run_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
    return redirect(url_for('index'))

@app.route('/upload_localized/<run_name>', methods=['POST'])
def upload_localized(run_name):
    # Define o diretório de destino dentro da pasta de outputs do experimento
    # Caminho: uploads/outputs/<run_name>/localized_custom_gs/
    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name, 'localized_custom_gs')
    os.makedirs(target_dir, exist_ok=True)

    ply_file = request.files.get('selected_ply')
    tex_file = request.files.get('texture_img')
    config_file = request.files.get('localized_custom_gs.yaml')

    if ply_file and tex_file and config_file:
        # Configurações para a execução do Docker de Retexturização
        run_config_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name)

        # Salva o YAML de configuração localizado na pasta de configs do run
        config_path = os.path.join(run_config_dir, 'localized_custom_gs.yaml')
        config_file.save(config_path)

        # Salva o PLY selecionado
        ply_path = os.path.join(target_dir, 'splats_selected.ply')
        ply_file.save(ply_path)
        
        # Salva a textura (preservando a extensão original)
        tex_ext = os.path.splitext(tex_file.filename)[1]
        tex_path = os.path.join(target_dir, 'external_texture' + tex_ext)
        
        tex_file.save(tex_path)

        abs_run_config_dir = os.path.abspath(run_config_dir) # Onde está o localized_custom_gs.yaml
        abs_host_output_dir = os.path.dirname(abs_run_config_dir) # Pai da pasta do experimento

        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f"{app.config['HOST_DATA_DIR']}:/data",
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_run_config_dir}:/app/configs',
            '--gpus', 'all',
            '--name', f'texture-gs-retexture-{run_name}',
            'texture-gs', 'retexture.sh'
        ]

        log_filename = f'{run_name}.txt'
        success = _run_docker_container(docker_command, log_filename, "Docker Retexture Command")

        if success:
            return render_template('index.html', log_link=url_for('get_log', filename=log_filename), logs=get_logs_list(), datasets=get_datasets_list())
        else:
            return f"Erro ao executar retexturização. Veja o log: <a href='{url_for('get_log', filename=log_filename)}'>{log_filename}</a>", 500

    return "Arquivos ausentes", 400

@app.route('/api/v1/retexture', methods=['POST'])
def api_upload_localized():
    """
    Endpoint de API para retexturização localizada.
    Espera: file_selected (PLY), file_texture (Imagem), run_name (Form Data)
    Assume que localized_custom_gs.yaml já existe na pasta do RUN_NAME.
    """
    run_name = request.form.get('run_name')
    if not run_name:
        return jsonify({"error": "O campo 'run_name' é obrigatório para identificar o experimento."}), 400
    
    run_name = secure_filename(run_name)
    run_config_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', run_name)
    
    # Verifica se o experimento base existe
    if not os.path.exists(run_config_dir):
        return jsonify({"error": f"Experimento '{run_name}' não encontrado. Certifique-se de que o RUN_NAME está correto."}), 404

    # Verifica se o arquivo de configuração obrigatório já está lá
    if not os.path.exists(os.path.join(run_config_dir, 'localized_custom_gs.yaml')):
        return jsonify({"error": f"Arquivo 'localized_custom_gs.yaml' não encontrado em {run_name}. Envie-o primeiro via interface ou garanta sua existência."}), 400

    ply_file = request.files.get('file_selected')
    tex_file = request.files.get('file_texture')

    if not (ply_file and tex_file):
        return jsonify({"error": "Arquivos 'file_selected' (PLY) e 'file_texture' (PNG/JPG) são obrigatórios."}), 400

    # Define diretório de destino para os novos inputs
    target_dir = os.path.join(run_config_dir, 'localized_custom_gs')
    os.makedirs(target_dir, exist_ok=True)

    # Salva os arquivos recebidos
    ply_file.save(os.path.join(target_dir, 'splats_selected.ply'))
    tex_ext = os.path.splitext(tex_file.filename)[1]
    tex_filename = 'external_texture' + tex_ext
    tex_file.save(os.path.join(target_dir, tex_filename))

    # Atualiza o parâmetro texture_filepath no YAML de configuração
    config_path = os.path.join(run_config_dir, 'localized_custom_gs.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if config_data and 'input' in config_data:
            config_data['input']['texture_filepath'] = f"output/{run_name}/localized_custom_gs/{tex_filename}"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)

    # Configurações para a execução do Docker
    abs_run_config_dir = os.path.abspath(run_config_dir)
    abs_host_output_dir = os.path.dirname(abs_run_config_dir)

    docker_command = [
        'docker', 'run', '--rm',
        '--user', f'{os.getuid()}:{os.getgid()}',
        '-e', f'RUN_NAME={run_name}',
        '-v', f"{app.config['HOST_DATA_DIR']}:/data",
        '-v', f'{abs_host_output_dir}:/app/output',
        '-v', f'{abs_run_config_dir}:/app/configs',
        '--gpus', 'all',
        '--name', f'texture-gs-retexture-api-{run_name}',
        'texture-gs', 'retexture.sh'
    ]

    log_filename = f'{run_name}.txt'
    success = _run_docker_container(docker_command, log_filename, "API Docker Retexture Command")

    if success:
        return jsonify({
            "status": "success",
            "message": "Processo de retexturização iniciado com sucesso via API.",
            "log_url": url_for('get_log', filename=log_filename, _external=True),
            "combined_ply_url": url_for('download_combined_ply', run_name=run_name, _external=True)
        }), 200
    else:
        return jsonify({"error": "Falha ao executar o container Docker. Verifique os logs."}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'outputs'), exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
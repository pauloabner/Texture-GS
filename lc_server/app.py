import os
import subprocess
import uuid
import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
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

ALLOWED_EXTENSIONS = {'yaml'}

def get_datasets_list():
    """Lista as pastas dentro do diretório de dados para o combo box."""
    data_path = '/mnt/abner/data_dtu'
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
        # log_filename = f'{run_name}_{run_id}.txt'
        try:
            name_without_ext = filename[:-4]
            parts = name_without_ext.rsplit('_', 1)
            if len(parts) == 2:
                run_name, run_id = parts
                # Caminho: uploads/<run_id>/outputs/<run_name>/texture_gaussian3d/pcds/40000.ply
                ply_path = os.path.join(app.config['UPLOAD_FOLDER'], run_id, 'outputs', run_name, 'texture_gaussian3d', 'pcds', '40000.ply')
                exists = os.path.exists(ply_path)
                logs_data.append({
                    'filename': filename,
                    'run_name': run_name,
                    'run_id': run_id,
                    'ply_exists': exists
                })
            else:
                logs_data.append({'filename': filename, 'ply_exists': False})
        except Exception:
            logs_data.append({'filename': filename, 'ply_exists': False})
    
    return logs_data

@app.route('/')
def index():
    return render_template('index.html', logs=get_logs_list())

@app.route('/datasets')
def datasets_page():
    return render_template('datasets.html', datasets=get_datasets_list(), logs=get_logs_list())

@app.route('/start_from_dataset', methods=['POST'])
def start_from_dataset():
    run_name = request.form.get('run_name', 'DEFAULT_EXPERIMENT')
    dataset_path = request.form.get('dataset_path')

    if not dataset_path:
        return "Nenhum dataset selecionado", 400

    # 1. Gerar um ID único para a execução e definir caminhos
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    
    # Pasta de destino conforme solicitado: outputs/<RUN_NAME>
    target_dir = os.path.join(run_dir, 'outputs', run_name)
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
        HOST_DATA_DIR = '/mnt/abner/data_dtu'
        abs_target_dir = os.path.abspath(target_dir) # Onde os YAMLs foram copiados
        abs_host_output_dir = os.path.abspath(os.path.join(run_dir, 'outputs'))

        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f'{HOST_DATA_DIR}:/data',
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_target_dir}:/app/configs', # Monta a pasta com os YAMLs processados
            '--gpus', 'all',
            '--name', f'texture-gs-{run_id}',
            'texture-gs'
        ]

        log_filename = f'{run_name}_{run_id}.txt'
        log_filepath = os.path.join(app.config['LOG_FOLDER'], log_filename)

        try:
            with open(log_filepath, 'w') as log_file:
                process = subprocess.run(
                    docker_command,
                    capture_output=True,
                    text=True,
                    check=False
                )
                log_file.write("--- Docker Command (from Dataset Page) ---\n")
                log_file.write(" ".join(docker_command) + "\n\n")
                log_file.write("--- Standard Output ---\n")
                log_file.write(process.stdout)
                log_file.write("\n--- Standard Error ---\n")
                log_file.write(process.stderr)
                log_file.write(f"\n--- Exit Code: {process.returncode} ---\n")
            
            return render_template(
                'datasets.html', 
                log_link=url_for('get_log', filename=log_filename),
                logs=get_logs_list(),
                datasets=get_datasets_list()
            )
        except Exception as e:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"\n--- Error during execution: {e} ---\n")
            return f"Erro ao executar Docker: {e}. Veja o log: <a href='{url_for('get_log', filename=log_filename)}'>{log_filename}</a>", 500

    return "Erro: Pasta de templates não encontrada", 500

@app.route('/run_texture_gs', methods=['POST'])
def run_texture_gs():
    if request.method == 'POST':
        run_name = request.form.get('run_name', 'DEFAULT_EXPERIMENT')
        
        # Create a unique directory for this run's configs
        run_id = str(uuid.uuid4())
        run_config_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
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
        HOST_DATA_DIR = '/mnt/abner/data_dtu'
        host_output_dir = os.path.join(run_config_dir, 'outputs')
        os.makedirs(host_output_dir, exist_ok=True)
        
        # Get the absolute path of the run_config_dir for Docker volume mounting
        abs_run_config_dir = os.path.abspath(run_config_dir)
        abs_host_output_dir = os.path.abspath(host_output_dir)

        # Construct the Docker command
        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f'{HOST_DATA_DIR}:/data',
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_run_config_dir}:/app/configs', # Mount the uploaded configs
            '--gpus', 'all',
            '--name', f'texture-gs-{run_id}', # Unique name for the container
            'texture-gs'
        ]

        log_filename = f'{run_name}_{run_id}.txt'
        log_filepath = os.path.join(app.config['LOG_FOLDER'], log_filename)

        try:
            with open(log_filepath, 'w') as log_file:
                process = subprocess.run(
                    docker_command,
                    capture_output=True,
                    text=True,
                    check=False # Do not raise an exception for non-zero exit codes
                )
                log_file.write("--- Docker Command ---\n")
                log_file.write(" ".join(docker_command) + "\n\n")
                log_file.write("--- Standard Output ---\n")
                log_file.write(process.stdout)
                log_file.write("\n--- Standard Error ---\n")
                log_file.write(process.stderr)
                log_file.write(f"\n--- Exit Code: {process.returncode} ---\n")
            
            # Comentado para depuração: os arquivos de configuração agora permanecem 
            # na pasta uploads/<run_id> para que você possa conferir o conteúdo.
            # for f in config_files_uploaded:
            #     if os.path.exists(f):
            #         os.remove(f)

            return render_template(
                'index.html', 
                log_link=url_for('get_log', filename=log_filename),
                logs=get_logs_list()
            )
        except Exception as e:
            with open(log_filepath, 'a') as log_file:
                log_file.write(f"\n--- Error during execution: {e} ---\n")
            return f"An error occurred: {e}. Check log for details: <a href='{url_for('get_log', filename=log_filename)}'>{log_filename}</a>", 500

@app.route('/logs/<filename>')
def get_log(filename):
    return send_from_directory(app.config['LOG_FOLDER'], filename)

@app.route('/download_ply/<run_id>/<run_name>')
def download_ply(run_id, run_name):
    # Reconstrói o caminho para o arquivo 40000.ply dentro da pasta de outputs do run_id
    directory = os.path.join(app.config['UPLOAD_FOLDER'], run_id, 'outputs', run_name, 'texture_gaussian3d', 'pcds')
    filename = '40000.ply'
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/delete_run/<run_id>/<filename>')
def delete_run(run_id, filename):
    # 1. Remove o arquivo de log
    log_path = os.path.join(app.config['LOG_FOLDER'], filename)
    if os.path.exists(log_path):
        os.remove(log_path)
    # 2. Remove a pasta de processamento (uploads/<run_id>)
    if run_id and run_id != 'none':
        run_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
    return redirect(url_for('index'))

@app.route('/upload_localized/<run_id>/<run_name>', methods=['POST'])
def upload_localized(run_id, run_name):
    # Define o diretório de destino dentro da pasta de outputs do experimento
    # Caminho: uploads/<run_id>/outputs/<run_name>/localized_custom_gs/
    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id, 'outputs', run_name, 'localized_custom_gs')
    os.makedirs(target_dir, exist_ok=True)

    ply_file = request.files.get('selected_ply')
    tex_file = request.files.get('texture_img')
    config_file = request.files.get('localized_custom_gs.yaml')

    if ply_file and tex_file and config_file:
        # Configurações para a execução do Docker de Retexturização
        run_config_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)

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

        abs_run_config_dir = os.path.abspath(run_config_dir)
        abs_host_output_dir = os.path.abspath(os.path.join(run_config_dir, 'outputs'))
        HOST_DATA_DIR = '/mnt/abner/data_dtu'

        docker_command = [
            'docker', 'run', '--rm',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-e', f'RUN_NAME={run_name}',
            '-v', f'{HOST_DATA_DIR}:/data',
            '-v', f'{abs_host_output_dir}:/app/output',
            '-v', f'{abs_run_config_dir}:/app/configs',
            '--gpus', 'all',
            '--name', f'texture-gs-retexture-{run_id}',
            'texture-gs', 'retexture.sh'
        ]

        log_filename = f'RETEXTURE_{run_name}_{run_id}.txt'
        log_filepath = os.path.join(app.config['LOG_FOLDER'], log_filename)

        try:
            with open(log_filepath, 'w') as log_file:
                process = subprocess.run(docker_command, capture_output=True, text=True, check=False)
                log_file.write("--- Docker Retexture Command ---\n")
                log_file.write(" ".join(docker_command) + "\n\n")
                log_file.write("--- Standard Output ---\n")
                log_file.write(process.stdout)
                log_file.write("\n--- Standard Error ---\n")
                log_file.write(process.stderr)
                log_file.write(f"\n--- Exit Code: {process.returncode} ---\n")
            
            return render_template(
                'index.html', 
                log_link=url_for('get_log', filename=log_filename), 
                logs=get_logs_list(),
                datasets=get_datasets_list()
            )
        except Exception as e:
            return f"Erro ao executar retexturização: {e}", 500

    return "Arquivos ausentes", 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
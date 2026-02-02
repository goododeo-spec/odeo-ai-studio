/**
 * ODEO AI Studio
 * 视频 LoRA 训练与推理平台
 */

// 配置将从后端 API 获取，这里只是默认值
const CONFIG = {
    datasetPath: './data/datasets',
    rawPath: './data/datasets/raw',
    outputPath: './data/outputs',
    modelPath: './pretrained_models/Wan2.1-I2V-14B-480P'
};

const State = {
    currentPage: localStorage.getItem('currentPage') || 'home',
    currentDataTab: 'raw',
    currentTaskId: null,
    tasks: [],
    rawVideos: [],
    processedVideos: [],
    selectedGpu: null,
    gpus: [],
    isProcessing: false,
    isUploading: false,
    previewVideo: null,
    lossHistory: [] // 用于 loss 曲线
};

// ============================================
// API
// ============================================
const API = {
    baseUrl: '/api/v1',
    
    async request(url, options = {}) {
        try {
            const res = await fetch(this.baseUrl + url, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options
            });
            return await res.json();
        } catch (e) {
            console.error('API Error:', e);
            return { code: 500, message: e.message };
        }
    },
    
    getTasks: () => API.request('/training/list'),
    getGpus: () => API.request('/gpu/status'),
    getTask: (id) => API.request(`/training/${id}`),
    createTask: async (data) => {
        try {
            const res = await fetch('/api/v1/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await res.json();
        } catch (e) {
            console.error('createTask error:', e);
            return { code: 500, message: e.message };
        }
    },
    getRawVideos: () => API.request('/preprocess/raw'),
    processVideos: (data) => API.request('/preprocess/videos', { method: 'POST', body: JSON.stringify(data) }),
    getProcessedVideos: () => API.request('/preprocess/list'),
    getArBuckets: () => API.request('/preprocess/ar-buckets'),
    updateCaption: (data) => API.request('/preprocess/caption', { method: 'PUT', body: JSON.stringify(data) }),
    deleteVideo: (filename) => API.request(`/preprocess/video/${filename}`, { method: 'DELETE' }),
    deleteRawVideo: (filename) => API.request(`/preprocess/raw/${filename}`, { method: 'DELETE' }),
    getFrame: (filename, frameNumber) => `/api/v1/preprocess/frame/${encodeURIComponent(filename)}?frame=${frameNumber}`,
    saveDraft: (data) => API.request('/training/draft', { method: 'POST', body: JSON.stringify(data) }),
    copyTask: (taskId, newName) => API.request('/training/copy', { method: 'POST', body: JSON.stringify({ task_id: taskId, new_name: newName }) })
};

// Frame buckets: 4n+1 from 1 to 121
const FRAME_BUCKETS = [];
for (let n = 0; 4*n + 1 <= 121; n++) {
    FRAME_BUCKETS.push(4*n + 1);
}

// ============================================
// 配置持久化
// ============================================
const DEFAULT_CONFIG = {
    epochs: 60,
    batch_size: 1,
    grad_accum: 1,
    grad_clip: 1.0,
    warmup: 20,
    save_epochs: 5,
    ckpt_epochs: 10,
    clip_mode: 'single_beginning',
    blocks_swap: 0,
    act_ckpt_mode: 'true',
    model_dtype: 'bfloat16',
    transformer_dtype: 'float8',
    lora_rank: 32,
    adapter_dtype: 'bfloat16',
    resolution: 480,
    ar_bucket: 'true',
    repeats: 5,
    optimizer: 'adamw_optimi',
    lr: '5e-5',
    betas: '0.9, 0.99',
    weight_decay: '0.01'
};

function saveLastConfig(config) {
    localStorage.setItem('lastTrainingConfig', JSON.stringify(config));
}

function loadLastConfig() {
    try {
        const saved = localStorage.getItem('lastTrainingConfig');
        return saved ? JSON.parse(saved) : DEFAULT_CONFIG;
    } catch {
        return DEFAULT_CONFIG;
    }
}

function applyConfigToForm(config) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    
    // 基础配置
    setSliderValue('epochs', cfg.epochs);
    setSliderValue('batch-size', cfg.batch_size);
    setSliderValue('grad-accum', cfg.grad_accum);
    document.getElementById('grad-clip')?.setAttribute('value', cfg.grad_clip);
    if (document.getElementById('grad-clip')) document.getElementById('grad-clip').value = cfg.grad_clip;
    setSliderValue('warmup', cfg.warmup);
    setSliderValue('save-epochs', cfg.save_epochs);
    setSliderValue('ckpt-epochs', cfg.ckpt_epochs);
    setSelectValue('clip-mode', cfg.clip_mode);
    setSliderValue('blocks-swap', cfg.blocks_swap);
    setSelectValue('act-ckpt-mode', cfg.act_ckpt_mode);
    
    // 模型配置
    setSelectValue('model-dtype', cfg.model_dtype);
    setSelectValue('transformer-dtype', cfg.transformer_dtype);
    setSliderValue('lora-rank', cfg.lora_rank);
    setSelectValue('adapter-dtype', cfg.adapter_dtype);
    
    // 数据配置
    setSelectValue('resolution', cfg.resolution);
    setSelectValue('ar-bucket', cfg.ar_bucket);
    setSliderValue('repeats', cfg.repeats);
    
    // 优化器
    setSelectValue('optimizer', cfg.optimizer);
    setInputValue('lr', cfg.lr);
    setInputValue('betas', cfg.betas);
    setInputValue('weight-decay', cfg.weight_decay);
}

function setSliderValue(id, value) {
    const slider = document.getElementById(id);
    const input = document.getElementById(`${id}-val`);
    if (slider) slider.value = value;
    if (input) input.value = value;
}

function setSelectValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

function getCurrentConfigFromForm() {
    return {
        epochs: parseInt(document.getElementById('epochs-val')?.value || 60),
        batch_size: parseInt(document.getElementById('batch-size-val')?.value || 1),
        grad_accum: parseInt(document.getElementById('grad-accum-val')?.value || 1),
        grad_clip: parseFloat(document.getElementById('grad-clip')?.value || 1.0),
        warmup: parseInt(document.getElementById('warmup-val')?.value || 20),
        save_epochs: parseInt(document.getElementById('save-epochs-val')?.value || 5),
        ckpt_epochs: parseInt(document.getElementById('ckpt-epochs-val')?.value || 10),
        clip_mode: document.getElementById('clip-mode')?.value || 'single_beginning',
        blocks_swap: parseInt(document.getElementById('blocks-swap-val')?.value || 0),
        act_ckpt_mode: document.getElementById('act-ckpt-mode')?.value || 'true',
        model_dtype: document.getElementById('model-dtype')?.value || 'bfloat16',
        transformer_dtype: document.getElementById('transformer-dtype')?.value || 'float8',
        lora_rank: parseInt(document.getElementById('lora-rank-val')?.value || 32),
        adapter_dtype: document.getElementById('adapter-dtype')?.value || 'bfloat16',
        resolution: parseInt(document.getElementById('resolution')?.value || 480),
        ar_bucket: document.getElementById('ar-bucket')?.value || 'true',
        repeats: parseInt(document.getElementById('repeats-val')?.value || 5),
        optimizer: document.getElementById('optimizer')?.value || 'adamw_optimi',
        lr: document.getElementById('lr')?.value || '5e-5',
        betas: document.getElementById('betas')?.value || '0.9, 0.99',
        weight_decay: document.getElementById('weight-decay')?.value || '0.01'
    };
}

// ============================================
// 生成日期时间前缀
// ============================================
function generateDatePrefix() {
    const now = new Date();
    const y = now.getFullYear().toString().slice(-2);
    const m = String(now.getMonth() + 1).padStart(2, '0');
    const d = String(now.getDate()).padStart(2, '0');
    const h = String(now.getHours()).padStart(2, '0');
    const min = String(now.getMinutes()).padStart(2, '0');
    return `${y}${m}${d}_${h}${min}_`;
}

// ============================================
// Toast
// ============================================
function toast(msg, type = 'info') {
    const c = document.getElementById('toast-container');
    if (!c) return;
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 3000);
}

// ============================================
// Page Navigation
// ============================================
function showHomePage() {
    document.getElementById('page-home').classList.add('active');
    document.getElementById('page-training').classList.remove('active');
    State.currentPage = 'home';
    localStorage.setItem('currentPage', 'home');
    loadTasks();
}

function showTrainingPage() {
    document.getElementById('page-home').classList.remove('active');
    document.getElementById('page-training').classList.add('active');
    State.currentPage = 'training';
    localStorage.setItem('currentPage', 'training');
    loadGpus();
    
    // 如果是编辑已有任务，加载任务数据
    if (State.currentTaskId) {
        loadTaskData(State.currentTaskId);
    } else {
        // 新建任务：加载上次配置，但清空视频和名称
        applyConfigToForm(loadLastConfig());
        document.getElementById('task-name').value = '';
        // 不加载视频，显示空状态
        State.rawVideos = [];
        State.processedVideos = [];
        updateRawVideoDisplay();
        updateProcessedDisplay();
    }
}

function restorePage() {
    const saved = localStorage.getItem('currentPage');
    if (saved === 'training' && State.currentTaskId) {
        showTrainingPage();
    } else {
        showHomePage();
    }
}

// ============================================
// Data Tabs
// ============================================
function switchDataTab(tab) {
    State.currentDataTab = tab;
    document.querySelectorAll('.data-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.data-content').forEach(c => c.classList.remove('active'));
    const tabBtn = document.querySelector(`.data-tab[data-tab="${tab}"]`);
    const tabContent = document.getElementById(`tab-${tab}`);
    if (tabBtn) tabBtn.classList.add('active');
    if (tabContent) tabContent.classList.add('active');
}

// ============================================
// Tasks
// ============================================
async function loadTasks() {
    const res = await API.getTasks();
    State.tasks = res.data?.tasks || [];
    renderTasks();
}

function renderTasks() {
    const grid = document.getElementById('task-grid');
    if (!grid) return;
    if (State.tasks.length === 0) {
        grid.innerHTML = '<div class="empty-hint">暂无训练任务，点击上方按钮开始</div>';
        return;
    }
    grid.innerHTML = State.tasks.map(task => `
        <div class="task-card" onclick="openTaskDetail('${task.task_id}')">
            <div class="task-thumb">
                <div class="placeholder">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="2" y="2" width="20" height="20" rx="2"/>
                        <path d="M7 2v20M17 2v20M2 12h20"/>
                    </svg>
                </div>
                <div class="task-badges">
                    <span class="badge ${task.status}" onclick="event.stopPropagation(); viewTaskStatus('${task.task_id}', '${(task.description || '训练任务').replace(/'/g, "\\'")}')">${getStatusText(task.status)}</span>
                    <span class="badge model">Wan</span>
                </div>
                <div class="task-menu" onclick="event.stopPropagation();">
                    <button class="menu-btn" onclick="toggleTaskMenu('${task.task_id}')">⋮</button>
                    <div class="menu-dropdown" id="menu-${task.task_id}">
                        <button onclick="copyTask('${task.task_id}', '${(task.description || '任务').replace(/'/g, "\\'")}')">复制任务</button>
                        <button onclick="deleteTask('${task.task_id}')">删除任务</button>
                    </div>
                </div>
            </div>
            <div class="task-info">
                <div class="task-name">${task.description || '待命名任务'}</div>
                <div class="task-date">${formatDate(task.created_at)}</div>
            </div>
        </div>
    `).join('');
}

function toggleTaskMenu(taskId) {
    // 关闭所有其他菜单
    document.querySelectorAll('.menu-dropdown.show').forEach(m => m.classList.remove('show'));
    const menu = document.getElementById(`menu-${taskId}`);
    if (menu) menu.classList.toggle('show');
}

// 点击其他地方关闭菜单
document.addEventListener('click', (e) => {
    if (!e.target.closest('.task-menu')) {
        document.querySelectorAll('.menu-dropdown.show').forEach(m => m.classList.remove('show'));
    }
});

async function copyTask(taskId, originalName) {
    const newName = prompt('请输入新任务名称:', generateDatePrefix() + originalName.replace(/^\d{6}_\d{4}_/, ''));
    if (!newName) return;
    
    try {
        const res = await API.copyTask(taskId, newName);
        if (res.code === 200 || res.code === 201) {
            toast('任务复制成功！', 'success');
            loadTasks();
        } else {
            throw new Error(res.message || '复制失败');
        }
    } catch (e) {
        toast(`复制失败: ${e.message}`, 'error');
    }
}

async function deleteTask(taskId) {
    if (!confirm('确定要删除此任务吗？')) return;
    try {
        const res = await API.request(`/training/${taskId}`, { method: 'DELETE' });
        if (res.code === 200) {
            toast('任务已删除', 'success');
            loadTasks();
        } else {
            throw new Error(res.message || '删除失败');
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

// 打开任务详情
function openTaskDetail(taskId) {
    State.currentTaskId = taskId;
    showTrainingPage();
}

// 加载任务数据并填充表单
async function loadTaskData(taskId) {
    const res = await API.getTask(taskId);
    if (res.code === 200 && res.data) {
        const task = res.data;
        
        // 填充任务名称
        document.getElementById('task-name').value = task.description || '';
        
        // 恢复配置
        if (task.config) {
            applyConfigToForm(task.config);
        }
        
        // 恢复GPU选择
        if (task.gpu_id !== undefined && task.gpu_id >= 0) {
            State.selectedGpu = task.gpu_id;
        }
        
        // 加载该任务的视频数据（如果有）
        if (task.raw_videos) {
            State.rawVideos = task.raw_videos.map(f => ({ filename: f }));
            updateRawVideoDisplay();
        } else {
            loadRawVideos();
        }
        
        if (task.processed_videos) {
            State.processedVideos = task.processed_videos;
            updateProcessedDisplay();
        } else {
            loadProcessedVideos();
        }
        
        // AR buckets
        if (task.dataset?.ar_buckets) {
            const arInput = document.getElementById('ar-buckets');
            if (arInput) arInput.value = task.dataset.ar_buckets.join(', ');
        }
        
        renderGpus();
    }
}

// 新建任务
function startNewTraining() {
    console.log('startNewTraining called');
    State.currentTaskId = null;
    State.rawVideos = [];
    State.processedVideos = [];
    State.selectedGpu = null;
    State.lossHistory = [];
    showTrainingPage();
}
window.startNewTraining = startNewTraining;

function viewTaskStatus(taskId, taskName) {
    showTrainingStatus(taskId, taskName);
}

function getStatusText(status) {
    const map = { 
        'queued': '排队中', 
        'running': '训练中', 
        'completed': '已完成', 
        'failed': '报错', 
        'stopped': '已停止',
        'paused': '已暂停',
        'draft': '草稿'
    };
    return map[status] || status;
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

// ============================================
// GPUs
// ============================================
async function loadGpus() {
    const res = await API.getGpus();
    State.gpus = res.data?.gpus || [];
    if (State.gpus.length === 0) {
        State.gpus = Array.from({ length: 8 }, (_, i) => ({
            gpu_id: i, name: `GPU ${i}`, status: 'available', memory: {}
        }));
    }
    renderGpus();
    renderGpuIndicators();
}

function renderGpus() {
    const grid = document.getElementById('gpu-grid');
    if (!grid) return;
    grid.innerHTML = State.gpus.map(gpu => {
        const mem = gpu.memory || {};
        // 优先使用 total_gb/free_gb，回退到 total/free (MB)
        const totalGB = mem.total_gb || (mem.total / 1024) || 0;
        const freeGB = mem.free_gb || (mem.free / 1024) || 0;
        const usedGB = (totalGB - freeGB).toFixed(1);
        const memPercent = mem.utilization || (totalGB > 0 ? Math.round((totalGB - freeGB) / totalGB * 100) : 0);
        const isAvailable = gpu.status === 'available';
        const statusClass = isAvailable ? 'available' : 'busy';
        return `
        <div class="gpu-item ${statusClass} ${State.selectedGpu === gpu.gpu_id ? 'selected' : ''}" 
             onclick="selectGpu(${gpu.gpu_id})">
            <div class="gpu-header">
                <span class="gpu-id">GPU ${gpu.gpu_id}</span>
                <span class="gpu-status-badge ${statusClass}">${isAvailable ? '空闲' : '训练中'}</span>
            </div>
            <div class="gpu-memory">
                <div class="memory-bar">
                    <div class="memory-used" style="width: ${memPercent}%"></div>
                </div>
                <div class="memory-text">${usedGB} / ${totalGB.toFixed(1)} GB</div>
            </div>
        </div>
    `}).join('');
}

function selectGpu(id) {
    const gpu = State.gpus.find(g => g.gpu_id === id);
    if (!gpu || (gpu.status !== 'available' && gpu.status !== 'AVAILABLE')) {
        toast('该GPU不可用', 'warning');
        return;
    }
    State.selectedGpu = id;
    renderGpus();
}

function renderGpuIndicators() {
    const c = document.getElementById('gpu-indicators');
    if (c) c.innerHTML = State.gpus.map(g => `<div class="gpu-dot ${g.status}"></div>`).join('');
}

// ============================================
// Raw Videos
// ============================================
async function loadRawVideos() {
    const res = await API.getRawVideos();
    State.rawVideos = res.data?.videos || [];
    updateRawVideoDisplay();
}

function updateRawVideoDisplay() {
    const zone = document.getElementById('upload-zone');
    const grid = document.getElementById('raw-video-grid');
    const countEl = document.getElementById('raw-count');
    
    if (countEl) countEl.textContent = State.rawVideos.length;
    
    if (State.rawVideos.length > 0) {
        if (zone) zone.style.display = 'none';
        if (grid) {
            grid.style.display = 'grid';
            renderRawVideos();
        }
    } else {
        if (zone) zone.style.display = 'flex';
        if (grid) grid.style.display = 'none';
    }
}

function renderRawVideos() {
    const grid = document.getElementById('raw-video-grid');
    if (!grid) return;
    grid.innerHTML = State.rawVideos.map(v => `
        <div class="raw-video-item">
            <video src="/api/v1/preprocess/raw/file/${encodeURIComponent(v.filename)}" 
                   muted loop onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;">
            </video>
            <div class="name">${v.filename}</div>
            <button class="remove-btn" onclick="event.stopPropagation(); removeRawVideo('${v.filename}')">×</button>
        </div>
    `).join('');
}

async function removeRawVideo(filename) {
    if (!confirm(`删除 ${filename}？`)) return;
    const res = await API.deleteRawVideo(filename);
    if (res.code === 200) {
        toast('已删除', 'success');
        loadRawVideos();
    } else {
        toast(`删除失败: ${res.message}`, 'error');
    }
}

async function handleAddVideos(files) {
    if (!files || files.length === 0) return;
    if (State.isUploading) return;
    
    const formData = new FormData();
    for (const file of files) {
        if (file.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {
            formData.append('files', file);
        }
    }
    
    const count = formData.getAll('files').length;
    if (count === 0) {
        toast('请选择视频文件', 'warning');
        return;
    }
    
    State.isUploading = true;
    showUploadingState(count);
    
    try {
        const res = await fetch('/api/v1/preprocess/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.code === 200) {
            toast(`已添加 ${data.data.count} 个视频`, 'success');
            await loadRawVideos();
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        toast(`上传失败: ${e.message}`, 'error');
    } finally {
        State.isUploading = false;
        hideUploadingState();
    }
    
    const input = document.getElementById('raw-file-input');
    if (input) input.value = '';
}

function showUploadingState(count) {
    const btn = document.querySelector('.panel-actions .btn-secondary');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span> 添加中 (${count})...`;
    }
}

function hideUploadingState() {
    const btn = document.querySelector('.panel-actions .btn-secondary');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M12 5v14M5 12h14"/></svg> 添加视频`;
    }
}

// ============================================
// Process Modal
// ============================================
function showProcessModal() {
    if (State.rawVideos.length === 0) {
        toast('请先添加原始视频', 'warning');
        return;
    }
    
    const modal = document.getElementById('process-modal');
    const select = document.getElementById('preview-video-select');
    const summaryCount = document.getElementById('summary-video-count');
    const framePreview = document.getElementById('frame-preview');
    
    if (!modal) return;
    
    if (select) {
        select.innerHTML = '<option value="">选择一个视频预览</option>' +
            State.rawVideos.map(v => `<option value="${v.filename}">${v.filename}</option>`).join('');
    }
    if (summaryCount) summaryCount.textContent = State.rawVideos.length;
    if (framePreview) framePreview.innerHTML = '<div class="frame-placeholder"><p>选择视频查看帧预览</p></div>';
    
    modal.style.display = 'flex';
}

function hideProcessModal() {
    const modal = document.getElementById('process-modal');
    if (modal) modal.style.display = 'none';
}

function loadVideoForPreview() {
    const select = document.getElementById('preview-video-select');
    const filename = select ? select.value : '';
    
    if (!filename) {
        const framePreview = document.getElementById('frame-preview');
        if (framePreview) framePreview.innerHTML = '<div class="frame-placeholder"><p>选择视频查看帧预览</p></div>';
        State.previewVideo = null;
        return;
    }
    
    State.previewVideo = filename;
    updateFramePreview();
}

function updateFramePreview() {
    if (!State.previewVideo) return;
    
    const slider = document.getElementById('modal-frame-number');
    const input = document.getElementById('modal-frame-val');
    const previewDiv = document.getElementById('frame-preview');
    
    const frameNumber = slider ? slider.value : 30;
    if (input) input.value = frameNumber;
    
    if (previewDiv) {
        const imgUrl = API.getFrame(State.previewVideo, frameNumber);
        previewDiv.innerHTML = `<img src="${imgUrl}" alt="帧 ${frameNumber}" onerror="this.parentElement.innerHTML='<div class=frame-placeholder><p>无法加载帧</p></div>'">`;
    }
}

async function confirmProcessing() {
    if (State.isProcessing) return;
    
    const triggerWord = document.getElementById('modal-trigger-word')?.value || '';
    const captionMethod = document.getElementById('modal-caption-method')?.value || 'qwen';
    const frameNumber = parseInt(document.getElementById('modal-frame-val')?.value || 30);
    const fps = parseInt(document.getElementById('modal-fps')?.value || 16);
    
    hideProcessModal();
    State.isProcessing = true;
    showProcessStatus('正在处理...');
    
    try {
        const res = await API.processVideos({
            input_dir: CONFIG.rawPath,
            output_dir: CONFIG.datasetPath,
            prompt_prefix: triggerWord,
            caption_method: captionMethod,
            frame_number: frameNumber,
            fps: fps,
            use_qwen: captionMethod === 'qwen'
        });
        
        if (res.code === 200 || res.code === 201) {
            hideProcessStatus();
            toast(`处理完成！${res.data?.processed?.length || 0} 个视频`, 'success');
            await loadProcessedVideos();
            switchDataTab('processed');
        } else {
            throw new Error(res.message || '处理失败');
        }
    } catch (e) {
        hideProcessStatus();
        toast(`处理失败: ${e.message}`, 'error');
    } finally {
        State.isProcessing = false;
    }
}

function showProcessStatus(text) {
    const status = document.getElementById('process-status');
    const statusText = document.getElementById('status-text');
    const btn = document.getElementById('btn-process');
    
    if (status) status.style.display = 'flex';
    if (statusText) statusText.textContent = text;
    if (btn) btn.disabled = true;
}

function hideProcessStatus() {
    const status = document.getElementById('process-status');
    const btn = document.getElementById('btn-process');
    
    if (status) status.style.display = 'none';
    if (btn) btn.disabled = false;
}

// ============================================
// Processed Videos
// ============================================
async function loadProcessedVideos() {
    const res = await API.getProcessedVideos();
    State.processedVideos = res.data?.videos || [];
    updateProcessedDisplay();
    loadArBuckets();
}

async function loadArBuckets() {
    try {
        const res = await API.getArBuckets();
        if (res.code === 200 && res.data?.ar_buckets?.length > 0) {
            const arInput = document.getElementById('ar-buckets');
            if (arInput) arInput.value = res.data.ar_buckets.join(', ');
        }
    } catch (e) {}
}

function updateProcessedDisplay() {
    const list = document.getElementById('processed-list');
    const empty = document.getElementById('processed-empty');
    const countEl = document.getElementById('processed-count');
    
    if (countEl) countEl.textContent = State.processedVideos.length;
    
    if (State.processedVideos.length > 0) {
        if (list) { list.style.display = 'block'; renderProcessedVideos(); }
        if (empty) empty.style.display = 'none';
    } else {
        if (list) list.style.display = 'none';
        if (empty) empty.style.display = 'flex';
    }
}

function renderProcessedVideos() {
    const list = document.getElementById('processed-list');
    if (!list) return;
    list.innerHTML = State.processedVideos.map((v, i) => `
        <div class="processed-card" data-id="${i}">
            <div class="card-video">
                <video src="/api/v1/preprocess/video/${v.filename}" muted loop onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;"></video>
                <div class="card-filename">${v.filename}</div>
            </div>
            <div class="card-caption">
                <textarea id="caption-${i}" onchange="markChanged(${i})" placeholder="输入提示词...">${v.caption || ''}</textarea>
            </div>
            <div class="card-actions">
                <button class="btn-icon" onclick="saveCaption(${i}, '${v.filename}')" title="保存">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17,21 17,13 7,13 7,21"/><polyline points="7,3 7,8 15,8"/></svg>
                </button>
                <button class="btn-icon btn-icon-danger" onclick="deleteProcessedVideo('${v.filename}')" title="删除">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="3,6 5,6 21,6"/><path d="M19,6v14a2,2 0 01-2,2H7a2,2 0 01-2-2V6m3,0V4a2,2 0 012-2h4a2,2 0 012,2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

function markChanged(id) {
    const item = document.querySelector(`.processed-card[data-id="${id}"]`);
    if (item) item.classList.add('changed');
}

async function saveCaption(id, filename) {
    const textarea = document.getElementById(`caption-${id}`);
    if (!textarea) return;
    try {
        const res = await API.updateCaption({ filename, caption: textarea.value });
        if (res.code === 200) {
            toast('已保存', 'success');
            const item = document.querySelector(`.processed-card[data-id="${id}"]`);
            if (item) item.classList.remove('changed');
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`保存失败: ${e.message}`, 'error');
    }
}

async function deleteProcessedVideo(filename) {
    if (!confirm(`删除 ${filename}？`)) return;
    try {
        const res = await API.deleteVideo(filename);
        if (res.code === 200) {
            toast('已删除', 'success');
            loadProcessedVideos();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

// ============================================
// Training
// ============================================
function getFormValues() {
    const actCkptMode = document.getElementById('act-ckpt-mode')?.value || 'true';
    let actCkpt = true;
    if (actCkptMode === 'false') actCkpt = false;
    else if (actCkptMode === 'unsloth') actCkpt = 'unsloth';
    
    return {
        output_dir: CONFIG.outputPath,
        epochs: parseInt(document.getElementById('epochs-val')?.value || 60),
        micro_batch_size_per_gpu: parseInt(document.getElementById('batch-size-val')?.value || 1),
        gradient_accumulation_steps: parseInt(document.getElementById('grad-accum-val')?.value || 1),
        gradient_clipping: parseFloat(document.getElementById('grad-clip')?.value || 1.0),
        warmup_steps: parseInt(document.getElementById('warmup-val')?.value || 20),
        save_every_n_epochs: parseInt(document.getElementById('save-epochs-val')?.value || 5),
        checkpoint_every_n_epochs: parseInt(document.getElementById('ckpt-epochs-val')?.value || 10),
        activation_checkpointing: actCkpt,
        blocks_to_swap: parseInt(document.getElementById('blocks-swap-val')?.value || 0),
        video_clip_mode: document.getElementById('clip-mode')?.value || 'single_beginning',
        model: {
            type: 'wan',
            ckpt_path: CONFIG.modelPath,
            dtype: document.getElementById('model-dtype')?.value || 'bfloat16',
            transformer_dtype: document.getElementById('transformer-dtype')?.value || 'float8'
        },
        adapter: {
            type: 'lora',
            rank: parseInt(document.getElementById('lora-rank-val')?.value || 32),
            dtype: document.getElementById('adapter-dtype')?.value || 'bfloat16'
        },
        optimizer: {
            type: document.getElementById('optimizer')?.value || 'adamw_optimi',
            lr: parseFloat(document.getElementById('lr')?.value || 5e-5),
            betas: document.getElementById('betas')?.value.split(',').map(v => parseFloat(v.trim())),
            weight_decay: parseFloat(document.getElementById('weight-decay')?.value || 0.01)
        },
        dataset_config: {
            resolutions: [parseInt(document.getElementById('resolution')?.value || 480)],
            enable_ar_bucket: document.getElementById('ar-bucket')?.value === 'true',
            ar_buckets: document.getElementById('ar-buckets')?.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
            frame_buckets: document.getElementById('frame-buckets')?.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
            directory: { path: CONFIG.datasetPath, num_repeats: parseInt(document.getElementById('repeats-val')?.value || 5) }
        }
    };
}

async function startTraining() {
    if (State.processedVideos.length === 0) {
        toast('请先处理训练数据', 'warning');
        switchDataTab('processed');
        return;
    }
    if (State.selectedGpu === null) {
        toast('请选择GPU', 'warning');
        return;
    }
    let name = document.getElementById('task-name')?.value?.trim();
    if (!name) {
        toast('请输入任务名称', 'warning');
        document.getElementById('task-name')?.focus();
        return;
    }
    
    // 自动添加日期前缀（如果没有的话）
    if (!/^\d{6}_\d{4}_/.test(name)) {
        name = generateDatePrefix() + name;
        document.getElementById('task-name').value = name;
    }
    
    const btn = document.getElementById('start-btn');
    if (btn) { btn.disabled = true; btn.textContent = '创建中...'; }
    
    try {
        const formValues = getFormValues();
        
        // 保存当前配置供下次使用
        saveLastConfig(getCurrentConfigFromForm());
        
        const res = await API.createTask({
            gpu_id: State.selectedGpu,
            model_type: 'wan',
            description: name,
            dataset: {
                resolutions: formValues.dataset_config.resolutions,
                enable_ar_bucket: formValues.dataset_config.enable_ar_bucket,
                ar_buckets: formValues.dataset_config.ar_buckets,
                frame_buckets: formValues.dataset_config.frame_buckets,
                directory: [{ path: CONFIG.datasetPath, num_repeats: formValues.dataset_config.directory.num_repeats }]
            },
            config: formValues,
            raw_videos: State.rawVideos.map(v => v.filename),
            processed_videos: State.processedVideos.map(v => ({ filename: v.filename, caption: v.caption }))
        });
        
        if (res.code === 201 || res.code === 200) {
            const taskId = res.data?.task_id;
            toast('训练任务创建成功！', 'success');
            State.lossHistory = [];
            if (taskId) showTrainingStatus(taskId, name);
            else setTimeout(() => showHomePage(), 1500);
        } else {
            throw new Error(res.message || '创建失败');
        }
    } catch (e) {
        toast(`创建失败: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '开始训练'; }
    }
}

// ============================================
// Save Draft
// ============================================
async function saveDraft() {
    let name = document.getElementById('task-name')?.value?.trim();
    if (!name) {
        toast('请输入任务名称', 'warning');
        document.getElementById('task-name')?.focus();
        return;
    }
    
    // 自动添加日期前缀
    if (!/^\d{6}_\d{4}_/.test(name)) {
        name = generateDatePrefix() + name;
        document.getElementById('task-name').value = name;
    }
    
    try {
        const formValues = getFormValues();
        saveLastConfig(getCurrentConfigFromForm());
        
        const draftData = {
            description: name,
            model_type: 'wan',
            gpu_id: State.selectedGpu,
            config: formValues,
            dataset: {
                resolutions: formValues.dataset_config.resolutions,
                enable_ar_bucket: formValues.dataset_config.enable_ar_bucket,
                ar_buckets: formValues.dataset_config.ar_buckets,
                frame_buckets: formValues.dataset_config.frame_buckets,
                directory: [{ path: CONFIG.datasetPath, num_repeats: formValues.dataset_config.directory.num_repeats }]
            },
            raw_videos: State.rawVideos.map(v => v.filename),
            processed_videos: State.processedVideos.map(v => ({ filename: v.filename, caption: v.caption }))
        };
        
        const res = await API.saveDraft(draftData);
        if (res.code === 201 || res.code === 200) {
            toast('草稿保存成功！', 'success');
            loadTasks();
        } else {
            throw new Error(res.message || '保存失败');
        }
    } catch (e) {
        toast(`保存失败: ${e.message}`, 'error');
    }
}

// ============================================
// Training Status Panel with Loss Chart
// ============================================
let statusPollInterval = null;

function showTrainingStatus(taskId, taskName) {
    State.currentTaskId = taskId;
    State.currentTaskName = taskName || '训练任务';
    State.lossHistory = [];
    
    let panel = document.getElementById('training-status-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'training-status-panel';
        panel.className = 'training-status-panel';
        document.body.appendChild(panel);
    }
    
    panel.innerHTML = `
        <div class="status-modal">
            <div class="status-header">
                <h3>${taskName || '训练任务'}</h3>
                <button class="close-btn" onclick="hideTrainingStatus()">×</button>
            </div>
            <div class="status-body">
                <div class="status-grid">
                    <div class="status-card">
                        <div class="card-label">状态</div>
                        <div class="card-value"><span class="status-badge queued" id="status-state">排队中</span></div>
                    </div>
                    <div class="status-card">
                        <div class="card-label">GPU</div>
                        <div class="card-value" id="status-gpu">-</div>
                    </div>
                    <div class="status-card">
                        <div class="card-label">Epoch</div>
                        <div class="card-value" id="status-epoch">0 / -</div>
                    </div>
                    <div class="status-card">
                        <div class="card-label">显存</div>
                        <div class="card-value" id="status-memory">-</div>
                    </div>
                </div>
                
                <div class="progress-section">
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="status-progress-bar" style="width: 0%"></div>
                    </div>
                    <div style="text-align: right; font-size: 12px; color: var(--text-muted); margin-top: 6px;">
                        <span id="status-progress">0%</span>
                    </div>
                </div>
                
                <div class="loss-chart-section">
                    <div class="chart-header">
                        <h4>Loss 曲线</h4>
                        <span class="current-loss" id="current-loss-value">当前: -</span>
                    </div>
                    <div class="loss-chart" id="loss-chart">
                        <canvas id="loss-canvas" width="540" height="160"></canvas>
                    </div>
                </div>
                
                <div class="logs-section">
                    <div class="logs-header">
                        <h4>训练日志</h4>
                        <span class="log-count" id="log-count">0 条</span>
                    </div>
                    <div class="log-container" id="status-logs">
                        <div class="log-placeholder">等待训练日志...</div>
                    </div>
                </div>
            </div>
            <div class="status-footer">
                <button class="btn-danger" onclick="stopTraining('${taskId}')">停止训练</button>
                <button class="btn-secondary" onclick="hideTrainingStatus()">关闭</button>
            </div>
        </div>
    `;
    
    panel.style.display = 'flex';
    startStatusPolling(taskId);
}

function hideTrainingStatus() {
    const panel = document.getElementById('training-status-panel');
    if (panel) panel.style.display = 'none';
    stopStatusPolling();
}

function startStatusPolling(taskId) {
    stopStatusPolling();
    fetchTrainingStatus(taskId);
    statusPollInterval = setInterval(() => fetchTrainingStatus(taskId), 2000);
}

function stopStatusPolling() {
    if (statusPollInterval) { clearInterval(statusPollInterval); statusPollInterval = null; }
}

async function fetchTrainingStatus(taskId) {
    try {
        const res = await API.request(`/training/${taskId}`);
        if (res.code === 200 && res.data) {
            updateTrainingStatusUI(res.data);
        } else if (res.code === 404) {
            stopStatusPolling();
            updateLogContainer([{ level: 'ERROR', message: '任务不存在或已被删除' }]);
        }
    } catch (e) {}
    
    try {
        const logsRes = await API.request(`/training/${taskId}/logs?tail=30`);
        if (logsRes.code === 200 && logsRes.data?.logs?.length > 0) {
            updateTrainingLogs(logsRes.data.logs);
        }
    } catch (e) {}
}

function updateLogContainer(logs) {
    const container = document.getElementById('status-logs');
    const countEl = document.getElementById('log-count');
    if (!container) return;
    
    if (!logs || logs.length === 0) {
        container.innerHTML = '<div class="log-placeholder">等待训练日志...</div>';
        if (countEl) countEl.textContent = '0 条';
        return;
    }
    
    container.innerHTML = logs.map(log => {
        const time = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '';
        const level = (log.level || 'INFO').toUpperCase();
        return `<div class="log-line ${level.toLowerCase()}">
            <span class="log-time">${time}</span>
            <span class="log-level">[${level}]</span>
            <span class="log-msg">${log.message || ''}</span>
        </div>`;
    }).join('');
    
    if (countEl) countEl.textContent = `${logs.length} 条`;
    container.scrollTop = container.scrollHeight;
}

function updateTrainingStatusUI(data) {
    const statusMap = { 'queued': '排队中', 'running': '训练中', 'completed': '已完成', 'failed': '失败', 'stopped': '已停止' };
    
    const stateEl = document.getElementById('status-state');
    const progressEl = document.getElementById('status-progress');
    const progressBar = document.getElementById('status-progress-bar');
    const epochEl = document.getElementById('status-epoch');
    const gpuEl = document.getElementById('status-gpu');
    const memoryEl = document.getElementById('status-memory');
    const lossValueEl = document.getElementById('current-loss-value');
    
    if (stateEl) {
        stateEl.textContent = statusMap[data.status] || data.status;
        stateEl.className = `status-badge ${data.status}`;
    }
    
    let progress = 0;
    if (data.progress && data.progress.total_epochs > 0) {
        progress = ((data.progress.current_epoch || 0) / data.progress.total_epochs) * 100;
    }
    if (progressEl) progressEl.textContent = `${progress.toFixed(1)}%`;
    if (progressBar) progressBar.style.width = `${progress}%`;
    
    if (epochEl && data.progress) epochEl.textContent = `${data.progress.current_epoch || 0} / ${data.progress.total_epochs || '-'}`;
    if (gpuEl) gpuEl.textContent = data.gpu_name || `GPU ${data.gpu_id || 0}`;
    
    if (memoryEl && data.system_stats?.gpu_memory) {
        const mem = data.system_stats.gpu_memory;
        memoryEl.textContent = `${mem.used || 0} / ${mem.total || 0} MB`;
    }
    
    // Loss 处理
    if (data.metrics?.current_loss != null) {
        const loss = data.metrics.current_loss;
        if (lossValueEl) lossValueEl.textContent = `当前: ${loss.toFixed(4)}`;
        
        // 添加到历史
        const step = data.progress?.current_step || State.lossHistory.length;
        if (State.lossHistory.length === 0 || State.lossHistory[State.lossHistory.length - 1].step !== step) {
            State.lossHistory.push({ step, loss });
            if (State.lossHistory.length > 100) State.lossHistory.shift();
            drawLossChart();
        }
    }
    
    if (['completed', 'failed', 'stopped'].includes(data.status)) {
        stopStatusPolling();
        if (data.status === 'completed') toast('训练完成！', 'success');
        else if (data.status === 'failed') toast(`训练失败: ${data.error_message || '未知错误'}`, 'error');
    }
}

function drawLossChart() {
    const canvas = document.getElementById('loss-canvas');
    if (!canvas || State.lossHistory.length < 2) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const padding = { top: 10, right: 10, bottom: 25, left: 50 };
    
    ctx.clearRect(0, 0, w, h);
    
    // 计算范围
    const losses = State.lossHistory.map(d => d.loss);
    const minLoss = Math.min(...losses) * 0.95;
    const maxLoss = Math.max(...losses) * 1.05;
    const minStep = State.lossHistory[0].step;
    const maxStep = State.lossHistory[State.lossHistory.length - 1].step;
    
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;
    
    const scaleX = (step) => padding.left + ((step - minStep) / (maxStep - minStep || 1)) * chartW;
    const scaleY = (loss) => padding.top + chartH - ((loss - minLoss) / (maxLoss - minLoss || 1)) * chartH;
    
    // 绘制网格
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
    }
    
    // 绘制坐标轴标签
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const val = maxLoss - (maxLoss - minLoss) * (i / 4);
        const y = padding.top + (chartH / 4) * i;
        ctx.fillText(val.toFixed(3), padding.left - 5, y + 3);
    }
    
    // 绘制曲线
    ctx.beginPath();
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    State.lossHistory.forEach((d, i) => {
        const x = scaleX(d.step);
        const y = scaleY(d.loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // 绘制渐变填充
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
    
    ctx.beginPath();
    State.lossHistory.forEach((d, i) => {
        const x = scaleX(d.step);
        const y = scaleY(d.loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.lineTo(scaleX(maxStep), h - padding.bottom);
    ctx.lineTo(scaleX(minStep), h - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // 绘制当前点
    if (State.lossHistory.length > 0) {
        const last = State.lossHistory[State.lossHistory.length - 1];
        ctx.beginPath();
        ctx.arc(scaleX(last.step), scaleY(last.loss), 4, 0, Math.PI * 2);
        ctx.fillStyle = '#6366f1';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

function updateTrainingLogs(logs) {
    updateLogContainer(logs);
}

async function stopTraining(taskId) {
    if (!confirm('确定要停止训练吗？')) return;
    
    try {
        const res = await API.request(`/training/stop/${taskId}`, { method: 'POST', body: JSON.stringify({ force: false }) });
        if (res.code === 200) {
            toast('训练已停止', 'success');
            stopStatusPolling();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`停止失败: ${e.message}`, 'error');
    }
}

// ============================================
// 推理功能
// ============================================
const InferState = {
    selectedTask: null,
    selectedLoras: [], // 支持多选
    testImagePath: null,
    testImageUrl: null,
    testImageWidth: 0,
    testImageHeight: 0,
    currentInferenceTask: null,
    trainingTasks: [],
    galleryFolders: [],
    galleryImages: [],
    selectedModalImages: [],
    currentGalleryFolder: ''
};

async function navigateTo(page) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
    
    document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
    document.getElementById(`page-${page}`)?.classList.add('active');
    
    localStorage.setItem('currentPage', page);
    
    if (page === 'inference') {
        await loadInferenceData();
    } else if (page === 'home') {
        loadTasks();
    } else if (page === 'gallery') {
        await loadGalleryData();
    }
}

async function loadInferenceData() {
    await Promise.all([
        loadTrainingTasksWithLoras(),
        loadInferenceGpus(),
        loadInferenceHistory()
    ]);
}

async function refreshInferenceData() {
    toast('刷新中...', 'info');
    await loadInferenceData();
    toast('刷新完成', 'success');
}

async function loadTrainingTasksWithLoras() {
    const container = document.getElementById('task-select-list');
    if (!container) return;
    
    container.innerHTML = '<div class="empty-state"><p>加载中...</p></div>';
    
    try {
        const res = await API.request('/inference/tasks-with-loras');
        if (res.code === 200 && res.data?.tasks) {
            InferState.trainingTasks = res.data.tasks;
            
            if (res.data.tasks.length === 0) {
                container.innerHTML = '<div class="empty-state"><p>暂无训练任务</p><p class="hint">请先完成模型训练</p></div>';
                return;
            }
            
            container.innerHTML = res.data.tasks.map(task => `
                <div class="task-select-item ${InferState.selectedTask === task.task_id ? 'selected' : ''}" 
                     onclick="selectTrainingTask('${task.task_id}')">
                    <div class="task-select-info">
                        <span class="task-select-name">${task.task_name || task.task_id.slice(-12)}</span>
                        <span class="task-select-meta">${task.epochs.length} 个版本 · 最新 epoch${task.latest_epoch}</span>
                    </div>
                </div>
            `).join('');
        } else {
            throw new Error(res.message || '获取失败');
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state error"><p>加载失败: ${e.message}</p></div>`;
    }
}

function selectTrainingTask(taskId) {
    InferState.selectedTask = taskId;
    InferState.selectedLoras = [];
    
    // 更新选中状态
    document.querySelectorAll('.task-select-item').forEach(el => el.classList.remove('selected'));
    document.querySelector(`.task-select-item[onclick*="${taskId}"]`)?.classList.add('selected');
    
    // 显示 epoch 选择区
    const epochSection = document.getElementById('epoch-select-section');
    if (epochSection) epochSection.style.display = 'block';
    
    // 渲染 epochs
    const task = InferState.trainingTasks.find(t => t.task_id === taskId);
    if (task) {
        renderEpochGrid(task.epochs);
    }
}

function renderEpochGrid(epochs) {
    const grid = document.getElementById('epoch-grid');
    if (!grid) return;
    
    grid.innerHTML = epochs.map(ep => `
        <div class="epoch-item ${InferState.selectedLoras.some(l => l.path === ep.path) ? 'selected' : ''}"
             onclick="toggleEpochSelection('${ep.path}', 'epoch${ep.epoch}', ${ep.epoch})">
            <span class="epoch-name">Epoch ${ep.epoch}</span>
            <span class="epoch-size">${ep.size_mb.toFixed(0)}MB</span>
        </div>
    `).join('');
    
    updateSelectedLorasTags();
}

function toggleEpochSelection(path, name, epoch) {
    const idx = InferState.selectedLoras.findIndex(l => l.path === path);
    if (idx >= 0) {
        InferState.selectedLoras.splice(idx, 1);
    } else {
        InferState.selectedLoras.push({ path, name, epoch });
    }
    
    // 更新 UI
    document.querySelectorAll('.epoch-item').forEach(el => {
        if (el.onclick.toString().includes(path)) {
            el.classList.toggle('selected', InferState.selectedLoras.some(l => l.path === path));
        }
    });
    
    updateSelectedLorasTags();
}

function updateSelectedLorasTags() {
    const container = document.getElementById('selected-loras');
    const tags = document.getElementById('selected-lora-tags');
    if (!container || !tags) return;
    
    if (InferState.selectedLoras.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'flex';
    tags.innerHTML = InferState.selectedLoras.map(l => `
        <span class="lora-tag">${l.name} <button onclick="removeLora('${l.path}')">×</button></span>
    `).join('');
}

function removeLora(path) {
    InferState.selectedLoras = InferState.selectedLoras.filter(l => l.path !== path);
    const task = InferState.trainingTasks.find(t => t.task_id === InferState.selectedTask);
    if (task) renderEpochGrid(task.epochs);
}

// 保留旧的 selectLora 兼容性
function selectLora(path, name) {
    InferState.selectedLoras = [{ path, name }];
    toast(`已选择 ${name}`, 'success');
}

async function loadInferenceGpus() {
    const select = document.getElementById('infer-gpu');
    if (!select) return;
    
    try {
        const res = await API.getGpus();
        if (res.code === 200 && res.data?.gpus) {
            select.innerHTML = res.data.gpus.map(gpu => 
                `<option value="${gpu.gpu_id}">GPU ${gpu.gpu_id} - ${gpu.memory?.free_gb?.toFixed(1) || '?'}GB 空闲</option>`
            ).join('');
        }
    } catch (e) {
        console.error('加载 GPU 失败:', e);
    }
}

// ============================================
// 图库弹窗
// ============================================
async function openGalleryModal() {
    // 重置选择状态
    InferState.selectedModalImages = [];
    document.getElementById('gallery-modal').style.display = 'flex';
    await loadModalGallery();
    updateSelectedCount();
}

function closeGalleryModal() {
    document.getElementById('gallery-modal').style.display = 'none';
}

async function loadModalGallery() {
    // 加载文件夹
    const folderList = document.getElementById('modal-folder-list');
    if (folderList) {
        try {
            const res = await API.request('/gallery/folders');
            if (res.code === 200) {
                let html = '<div class="folder-item active" data-folder="" onclick="selectModalFolder(\'\')"><span>全部图片</span></div>';
                res.data.folders.forEach(f => {
                    html += `<div class="folder-item" data-folder="${f.name}" onclick="selectModalFolder('${f.name}')">
                        <span>${f.name} (${f.count})</span>
                    </div>`;
                });
                folderList.innerHTML = html;
            }
        } catch (e) {
            console.error('加载文件夹失败:', e);
        }
    }
    
    // 加载图片
    await loadModalImages('');
}

async function selectModalFolder(folder) {
    InferState.currentGalleryFolder = folder;
    document.querySelectorAll('#modal-folder-list .folder-item').forEach(el => {
        el.classList.toggle('active', el.dataset.folder === folder);
    });
    await loadModalImages(folder);
}

async function loadModalImages(folder) {
    const grid = document.getElementById('modal-gallery-grid');
    if (!grid) return;
    
    grid.innerHTML = '<div class="empty-state"><p>加载中...</p></div>';
    
    try {
        const url = folder ? `/gallery/images?folder=${encodeURIComponent(folder)}` : '/gallery/images';
        const res = await API.request(url);
        
        if (res.code === 200 && res.data?.images) {
            if (res.data.images.length === 0) {
                grid.innerHTML = '<div class="empty-state"><p>暂无图片</p></div>';
                return;
            }
            
            grid.innerHTML = res.data.images.map(img => {
                const isSelected = InferState.selectedModalImages.some(s => s.id === img.id);
                return `
                <div class="gallery-image-item ${isSelected ? 'selected' : ''}"
                     data-id="${img.id}"
                     onclick="selectModalImage('${img.id}', '${img.path}', '${img.url}', event)">
                    <img src="${img.url}" alt="" loading="lazy">
                    <div class="select-indicator"></div>
                </div>
            `}).join('');
        }
    } catch (e) {
        grid.innerHTML = `<div class="empty-state error"><p>加载失败: ${e.message}</p></div>`;
    }
}

function selectModalImage(id, path, url, event) {
    // 多选模式 - 点击切换选中状态
    const idx = InferState.selectedModalImages.findIndex(img => img.id === id);
    if (idx >= 0) {
        // 已选中，取消选择
        InferState.selectedModalImages.splice(idx, 1);
    } else {
        // 未选中，添加选择
        InferState.selectedModalImages.push({ id, path, url });
    }
    
    // 更新 UI
    const el = event?.target?.closest('.gallery-image-item');
    if (el) {
        el.classList.toggle('selected', idx < 0);
    }
    
    // 更新选中计数
    updateSelectedCount();
}

function updateSelectedCount() {
    const countEl = document.getElementById('selected-image-count');
    if (countEl) {
        const count = InferState.selectedModalImages.length;
        countEl.textContent = count > 0 ? `已选择 ${count} 张` : '';
        countEl.style.display = count > 0 ? 'block' : 'none';
    }
}

async function confirmImageSelection() {
    if (InferState.selectedModalImages.length === 0) {
        toast('请选择至少一张图片', 'warning');
        return;
    }
    
    // 保存选中的图片列表
    InferState.testImages = [...InferState.selectedModalImages];
    
    // 使用第一张图片的尺寸作为参考
    const firstImage = InferState.selectedModalImages[0];
    InferState.testImagePath = firstImage.path;
    InferState.testImageUrl = firstImage.url;
    
    // 获取图片信息（尺寸）
    try {
        const res = await API.request(`/gallery/images/${encodeURIComponent(firstImage.id)}/info`);
        if (res.code === 200 && res.data) {
            InferState.testImageWidth = res.data.width;
            InferState.testImageHeight = res.data.height;
            
            // 自动设置宽高（按16倍数取整）
            const w = Math.round(res.data.width / 16) * 16;
            const h = Math.round(res.data.height / 16) * 16;
            document.getElementById('infer-width').value = w;
            document.getElementById('infer-height').value = h;
            
            // 显示尺寸信息
            const dimEl = document.getElementById('image-dimensions');
            const infoEl = document.getElementById('image-info');
            if (dimEl && infoEl) {
                dimEl.textContent = `${res.data.width} × ${res.data.height}`;
                infoEl.style.display = 'block';
            }
        }
    } catch (e) {
        console.error('获取图片信息失败:', e);
    }
    
    // 更新预览 - 显示多张图片的缩略图
    const preview = document.getElementById('test-image-preview');
    if (preview) {
        if (InferState.selectedModalImages.length === 1) {
            preview.innerHTML = `<img src="${firstImage.url}" alt="测试图片">`;
        } else {
            // 多张图片显示网格预览
            preview.innerHTML = `
                <div class="multi-preview">
                    ${InferState.selectedModalImages.slice(0, 4).map(img => 
                        `<img src="${img.url}" alt="">`
                    ).join('')}
                    ${InferState.selectedModalImages.length > 4 ? 
                        `<div class="more-count">+${InferState.selectedModalImages.length - 4}</div>` : ''}
                </div>
            `;
        }
    }
    
    closeGalleryModal();
    toast(`已选择 ${InferState.selectedModalImages.length} 张图片`, 'success');
}

async function handleModalUpload(files) {
    if (!files || files.length === 0) return;
    
    const folder = InferState.currentGalleryFolder || 'default';
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('folder', folder);
        
        try {
            const res = await fetch('/api/v1/gallery/images', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.code !== 200) throw new Error(data.message);
        } catch (e) {
            toast(`上传失败: ${e.message}`, 'error');
        }
    }
    
    toast('上传完成', 'success');
    await loadModalImages(folder);
}

async function handleTestImageUpload(files) {
    if (!files || files.length === 0) return;
    
    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('folder', 'uploads');
    
    try {
        const res = await fetch('/api/v1/gallery/images', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (data.code === 200) {
            InferState.testImagePath = data.data.path;
            InferState.testImageUrl = data.data.url;
            
            const preview = document.getElementById('test-image-preview');
            if (preview) preview.innerHTML = `<img src="${data.data.url}" alt="测试图片">`;
            
            // 获取尺寸
            const infoRes = await API.request(`/gallery/images/${encodeURIComponent(data.data.id)}/info`);
            if (infoRes.code === 200) {
                const w = Math.round(infoRes.data.width / 16) * 16;
                const h = Math.round(infoRes.data.height / 16) * 16;
                document.getElementById('infer-width').value = w;
                document.getElementById('infer-height').value = h;
            }
            
            toast('图片上传成功', 'success');
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        toast(`上传失败: ${e.message}`, 'error');
    }
}

async function startInference() {
    if (InferState.selectedLoras.length === 0) {
        toast('请选择 LoRA 模型', 'warning');
        return;
    }
    
    const prompt = document.getElementById('infer-prompt')?.value?.trim();
    if (!prompt) {
        toast('请输入提示词', 'warning');
        return;
    }
    
    // 使用第一个选中的 LoRA（后续可扩展为多 LoRA 融合）
    const lora = InferState.selectedLoras[0];
    
    const data = {
        lora_path: lora.path,
        lora_name: lora.name,
        prompt: prompt,
        image_path: InferState.testImagePath,
        width: parseInt(document.getElementById('infer-width')?.value) || 832,
        height: parseInt(document.getElementById('infer-height')?.value) || 480,
        num_frames: parseInt(document.getElementById('infer-frames')?.value) || 81,
        num_steps: parseInt(document.getElementById('infer-steps')?.value) || 30,
        guidance_scale: parseFloat(document.getElementById('infer-guidance')?.value) || 5.0,
        lora_strength: parseFloat(document.getElementById('infer-lora-strength')?.value) || 0.8,
        gpu_id: parseInt(document.getElementById('infer-gpu')?.value) || 0,
        seed: parseInt(document.getElementById('infer-seed')?.value) || -1
    };
    
    toast('正在创建推理任务...', 'info');
    
    try {
        const res = await API.request('/inference/create', {
            method: 'POST',
            body: JSON.stringify(data)
        });
        
        if (res.code === 201 && res.data) {
            InferState.currentInferenceTask = res.data.task_id;
            toast('推理任务已创建', 'success');
            
            pollInferenceProgress(res.data.task_id);
            updateResultArea('loading', res.data);
        } else {
            throw new Error(res.message || '创建失败');
        }
    } catch (e) {
        toast(`创建失败: ${e.message}`, 'error');
    }
}

async function pollInferenceProgress(taskId) {
    const poll = async () => {
        try {
            const res = await API.request(`/inference/task/${taskId}`);
            if (res.code === 200 && res.data) {
                const task = res.data;
                
                updateResultArea(task.status, task);
                
                if (task.status === 'running' || task.status === 'loading' || task.status === 'queued') {
                    setTimeout(poll, 2000);
                } else if (task.status === 'completed') {
                    toast('推理完成！', 'success');
                    loadInferenceHistory();
                } else if (task.status === 'failed') {
                    toast(`推理失败: ${task.error_message}`, 'error');
                }
            }
        } catch (e) {
            console.error('轮询失败:', e);
        }
    };
    
    poll();
}

function updateResultArea(status, task) {
    const area = document.getElementById('result-video-area');
    if (!area) return;
    
    if (status === 'loading' || status === 'queued') {
        area.innerHTML = `
            <div class="inference-progress">
                <div class="spinner"></div>
                <p>加载模型中...</p>
            </div>
        `;
    } else if (status === 'running') {
        const progress = task.progress || 0;
        area.innerHTML = `
            <div class="inference-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progress}%"></div>
                </div>
                <p>推理中... ${progress.toFixed(1)}%</p>
            </div>
        `;
    } else if (status === 'completed') {
        area.innerHTML = `
            <div class="result-success">
                <video controls autoplay loop>
                    <source src="/api/v1/inference/output/${task.task_id}/video" type="video/mp4">
                </video>
                <div class="result-info">
                    <span>任务 ID: ${task.task_id}</span>
                    <a href="/api/v1/inference/output/${task.task_id}/video" download class="btn-secondary">下载视频</a>
                </div>
            </div>
        `;
    } else if (status === 'failed') {
        area.innerHTML = `
            <div class="inference-error">
                <p>推理失败</p>
                <p class="error-msg">${task.error_message || '未知错误'}</p>
            </div>
        `;
    }
}

async function loadInferenceHistory() {
    const container = document.getElementById('inference-history-list');
    if (!container) return;
    
    try {
        const res = await API.request('/inference/tasks?limit=10');
        if (res.code === 200 && res.data?.tasks) {
            const tasks = res.data.tasks;
            if (tasks.length === 0) {
                container.innerHTML = '<div class="empty-state small"><p>暂无历史记录</p></div>';
                return;
            }
            
            container.innerHTML = tasks.map(task => `
                <div class="history-item ${task.status}" onclick="viewInferenceResult('${task.task_id}')">
                    <div class="history-info">
                        <span class="history-prompt">${(task.prompt || '').substring(0, 30)}...</span>
                        <span class="history-meta">${task.lora_name || ''} · ${formatDate(task.created_at)}</span>
                    </div>
                    <span class="history-status ${task.status}">${getInferenceStatusText(task.status)}</span>
                </div>
            `).join('');
        }
    } catch (e) {
        console.error('加载历史失败:', e);
    }
}

function getInferenceStatusText(status) {
    const map = {
        queued: '排队中',
        loading: '加载中',
        running: '推理中',
        completed: '已完成',
        failed: '失败'
    };
    return map[status] || status;
}

async function viewInferenceResult(taskId) {
    try {
        const res = await API.request(`/inference/task/${taskId}`);
        if (res.code === 200 && res.data) {
            updateResultArea(res.data.status, res.data);
        }
    } catch (e) {
        toast(`获取结果失败: ${e.message}`, 'error');
    }
}

// ============================================
// 图库页面功能
// ============================================
async function loadGalleryData() {
    await Promise.all([
        loadGalleryFolders(),
        loadGalleryImages('')
    ]);
}

async function loadGalleryFolders() {
    const container = document.getElementById('folder-list');
    if (!container) return;
    
    try {
        const res = await API.request('/gallery/folders');
        if (res.code === 200) {
            let html = `<div class="folder-item ${InferState.currentGalleryFolder === '' ? 'active' : ''}" data-folder="" onclick="selectGalleryFolder('')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                </svg>
                <span>全部图片</span>
            </div>`;
            
            res.data.folders.forEach(f => {
                html += `<div class="folder-item ${InferState.currentGalleryFolder === f.name ? 'active' : ''}" data-folder="${f.name}" onclick="selectGalleryFolder('${f.name}')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                    </svg>
                    <span>${f.name} (${f.count})</span>
                    <button class="folder-delete" onclick="event.stopPropagation(); deleteGalleryFolder('${f.name}')">×</button>
                </div>`;
            });
            container.innerHTML = html;
        }
    } catch (e) {
        console.error('加载文件夹失败:', e);
    }
}

async function selectGalleryFolder(folder) {
    InferState.currentGalleryFolder = folder;
    document.querySelectorAll('#folder-list .folder-item').forEach(el => {
        el.classList.toggle('active', el.dataset.folder === folder);
    });
    await loadGalleryImages(folder);
}

async function loadGalleryImages(folder) {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;
    
    grid.innerHTML = '<div class="empty-state"><p>加载中...</p></div>';
    
    try {
        const url = folder ? `/gallery/images?folder=${encodeURIComponent(folder)}` : '/gallery/images';
        const res = await API.request(url);
        
        if (res.code === 200 && res.data?.images) {
            if (res.data.images.length === 0) {
                grid.innerHTML = `<div class="empty-state">
                    <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5" width="64" height="64">
                        <rect x="8" y="8" width="48" height="48" rx="4"/>
                        <circle cx="20" cy="20" r="4"/>
                        <polyline points="56 40 40 24 16 48"/>
                    </svg>
                    <p>暂无图片</p>
                    <p class="hint">点击上方按钮上传图片</p>
                </div>`;
                return;
            }
            
            grid.innerHTML = res.data.images.map(img => `
                <div class="gallery-card">
                    <img src="${img.url}" alt="${img.name}" loading="lazy">
                    <div class="gallery-card-info">
                        <span class="name">${img.name}</span>
                        <button class="delete-btn" onclick="deleteGalleryImage('${img.id}')">删除</button>
                    </div>
                </div>
            `).join('');
        }
    } catch (e) {
        grid.innerHTML = `<div class="empty-state error"><p>加载失败: ${e.message}</p></div>`;
    }
}

async function createGalleryFolder() {
    const name = prompt('请输入文件夹名称:');
    if (!name) return;
    
    try {
        const res = await API.request('/gallery/folders', {
            method: 'POST',
            body: JSON.stringify({ name })
        });
        
        if (res.code === 201) {
            toast('文件夹创建成功', 'success');
            await loadGalleryFolders();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`创建失败: ${e.message}`, 'error');
    }
}

async function deleteGalleryFolder(name) {
    if (!confirm(`确定删除文件夹 "${name}" 及其中所有图片？`)) return;
    
    try {
        const res = await API.request(`/gallery/folders/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (res.code === 200) {
            toast('文件夹已删除', 'success');
            if (InferState.currentGalleryFolder === name) {
                InferState.currentGalleryFolder = '';
            }
            await loadGalleryData();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

function uploadToGallery() {
    document.getElementById('gallery-upload-input').click();
}

async function handleGalleryUpload(files) {
    if (!files || files.length === 0) return;
    
    const folder = InferState.currentGalleryFolder || 'default';
    toast(`正在上传 ${files.length} 个文件...`, 'info');
    
    let success = 0;
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('folder', folder);
        
        try {
            const res = await fetch('/api/v1/gallery/images', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.code === 200) success++;
        } catch (e) {
            console.error('上传失败:', e);
        }
    }
    
    toast(`上传完成: ${success}/${files.length}`, success === files.length ? 'success' : 'warning');
    await loadGalleryImages(folder);
}

async function deleteGalleryImage(imageId) {
    if (!confirm('确定删除这张图片？')) return;
    
    try {
        const res = await API.request(`/gallery/images/${encodeURIComponent(imageId)}`, { method: 'DELETE' });
        if (res.code === 200) {
            toast('图片已删除', 'success');
            await loadGalleryImages(InferState.currentGalleryFolder);
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

// ============================================
// Init
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing ODEO AI Studio...');
    
    // Config tabs
    document.querySelectorAll('.config-panel .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const panel = tab.dataset.tab;
            document.querySelectorAll('.config-panel .tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            const panelEl = document.getElementById(`panel-${panel}`);
            if (panelEl) panelEl.classList.add('active');
        });
    });
    
    // Sliders sync
    document.querySelectorAll('.slider-row').forEach(row => {
        const slider = row.querySelector('.slider');
        const input = row.querySelector('.slider-input');
        if (slider && input) {
            slider.addEventListener('input', () => input.value = slider.value);
            input.addEventListener('change', () => slider.value = input.value);
        }
    });
    
    // Modal frame slider sync
    const frameSlider = document.getElementById('modal-frame-number');
    const frameInput = document.getElementById('modal-frame-val');
    if (frameSlider && frameInput) {
        frameSlider.addEventListener('input', () => { frameInput.value = frameSlider.value; updateFramePreview(); });
        frameInput.addEventListener('change', () => { frameSlider.value = frameInput.value; updateFramePreview(); });
    }
    
    // Drag & drop
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
        uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); handleAddVideos(e.dataTransfer.files); });
    }
    
    // Restore
    restorePage();
    
    console.log('Init complete');
});

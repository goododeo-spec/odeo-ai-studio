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
    lossHistory: [], // 用于 loss 曲线
    isDirty: false,  // 是否有未保存的更改
    isNewTask: false // 是否是新建任务
};

// 生成任务ID
function generateTaskId() {
    const now = new Date();
    const timestamp = now.getFullYear().toString() +
        String(now.getMonth() + 1).padStart(2, '0') +
        String(now.getDate()).padStart(2, '0') + '_' +
        String(now.getHours()).padStart(2, '0') +
        String(now.getMinutes()).padStart(2, '0') +
        String(now.getSeconds()).padStart(2, '0');
    return `draft_${timestamp}`;
}

// 生成带时间前缀的任务名称
function generateTaskName() {
    const now = new Date();
    const prefix = String(now.getMonth() + 1).padStart(2, '0') +
        String(now.getDate()).padStart(2, '0') + '_' +
        String(now.getHours()).padStart(2, '0') +
        String(now.getMinutes()).padStart(2, '0');
    return prefix + '_';
}

// 标记有未保存的更改
function markDirty() {
    State.isDirty = true;
}

// 清除脏标记
function clearDirty() {
    State.isDirty = false;
}

// ============================================
// 状态持久化
// ============================================
const STORAGE_KEY = 'odeo_app_state';

function saveAppState() {
    try {
        const state = {
            // 页面状态
            currentPage: State.currentPage,
            currentDataTab: State.currentDataTab,
            currentTaskId: State.currentTaskId,
            isNewTask: State.isNewTask,
            
            // 推理页面状态
            inferState: {
                selectedTask: InferState?.selectedTask || null,
                selectedLoras: InferState?.selectedLoras || [],
                testImages: InferState?.testImages || [],
                testImagePath: InferState?.testImagePath || null,
                testImageUrl: InferState?.testImageUrl || null,
                testImageWidth: InferState?.testImageWidth || 0,
                testImageHeight: InferState?.testImageHeight || 0,
                currentGalleryFolder: InferState?.currentGalleryFolder || ''
            },
            
            // 推理参数表单
            inferParams: {
                prompt: document.getElementById('infer-prompt')?.value || '',
                loraStrength: document.getElementById('infer-lora-strength')?.value || '1',
                width: document.getElementById('infer-width')?.value || '832',
                height: document.getElementById('infer-height')?.value || '480',
                frames: document.getElementById('infer-frames')?.value || '81',
                steps: document.getElementById('infer-steps')?.value || '4',
                guidance: document.getElementById('infer-guidance')?.value || '1',
                seed: document.getElementById('infer-seed')?.value || '-1',
                autoCaption: document.getElementById('infer-auto-caption')?.checked ?? true
            },
            
            // 训练页面配置 tabs
            configTab: document.querySelector('.config-panel .tab.active')?.dataset?.tab || 'basic',
            
            // 触发词
            triggerWord: document.getElementById('trigger-word')?.value || '',
            
            // 训练配置参数
            trainingConfig: getCurrentConfigFromForm(),
            
            // 任务名称
            taskName: document.getElementById('task-name')?.value || '',
            
            // 保存时间戳
            savedAt: Date.now()
        };
        
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
        console.warn('保存状态失败:', e);
    }
}

function loadSavedState() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            return JSON.parse(saved);
        }
    } catch (e) {
        console.warn('加载状态失败:', e);
    }
    return null;
}

async function restoreAppState() {
    const saved = loadSavedState();
    if (!saved) {
        showHomePage();
        return;
    }
    
    // 恢复页面状态
    State.currentPage = saved.currentPage || 'home';
    State.currentDataTab = saved.currentDataTab || 'raw';
    State.currentTaskId = saved.currentTaskId || null;
    State.isNewTask = saved.isNewTask || false;
    
    // 恢复推理状态
    if (saved.inferState && typeof InferState !== 'undefined') {
        Object.assign(InferState, saved.inferState);
    }
    
    // 根据保存的页面类型恢复
    const page = saved.currentPage;
    
    if (page === 'training') {
        showTrainingPage();
    } else if (page === 'home') {
        showHomePage();
    } else if (page === 'inference' || page === 'gallery') {
        await navigateTo(page);
        if (page === 'inference' && saved.inferParams) {
            restoreInferParams(saved.inferParams);
        }
        if (page === 'inference' && saved.inferState) {
            await restoreInferenceState(saved.inferState);
        }
    } else {
        showHomePage();
    }
    
    // 不再从 localStorage 覆盖触发词
    // 非新建任务时由 loadTaskData 从 API 加载正确的触发词
    // 新建任务时已在 showTrainingPage 中清空
    
    // 恢复任务名称（在 loadTaskData 之后覆盖，仅对新任务有效）
    if (saved.taskName && State.isNewTask) {
        const tn = document.getElementById('task-name');
        if (tn) tn.value = saved.taskName;
    }
    
    // 恢复训练配置（如果不是从后端加载的任务，用 localStorage 的配置覆盖）
    if (saved.trainingConfig && State.isNewTask) {
        applyConfigToForm(saved.trainingConfig);
    }
    
    // 恢复 data tab（已处理 / 原始视频）
    if (saved.currentDataTab) {
        setTimeout(() => switchDataTab(saved.currentDataTab), 200);
    }
    
    // 恢复 config tab
    if (saved.configTab) {
        setTimeout(() => {
            const tab = document.querySelector(`.config-panel .tab[data-tab="${saved.configTab}"]`);
            if (tab) tab.click();
        }, 100);
    }
}

function restoreInferParams(params) {
    // 不恢复 prompt/trigger_word，由 selectTrainingTask 或 restoreInferenceState 根据选中任务自动填入
    if (params.loraStrength && document.getElementById('infer-lora-strength')) {
        document.getElementById('infer-lora-strength').value = params.loraStrength;
    }
    if (params.width && document.getElementById('infer-width')) {
        document.getElementById('infer-width').value = params.width;
    }
    if (params.height && document.getElementById('infer-height')) {
        document.getElementById('infer-height').value = params.height;
    }
    if (params.frames && document.getElementById('infer-frames')) {
        document.getElementById('infer-frames').value = params.frames;
    }
    if (params.steps && document.getElementById('infer-steps')) {
        document.getElementById('infer-steps').value = params.steps;
    }
    if (params.guidance && document.getElementById('infer-guidance')) {
        document.getElementById('infer-guidance').value = params.guidance;
    }
    if (params.seed && document.getElementById('infer-seed')) {
        document.getElementById('infer-seed').value = params.seed;
    }
    if (typeof params.autoCaption === 'boolean' && document.getElementById('infer-auto-caption')) {
        document.getElementById('infer-auto-caption').checked = params.autoCaption;
    }
}

async function restoreInferenceState(inferState) {
    // 恢复 InferState
    if (inferState.selectedTask) {
        InferState.selectedTask = inferState.selectedTask;
    }
    if (inferState.selectedLoras) {
        InferState.selectedLoras = inferState.selectedLoras;
    }
    if (inferState.testImages) {
        InferState.testImages = inferState.testImages;
    }
    if (inferState.testImagePath) {
        InferState.testImagePath = inferState.testImagePath;
    }
    if (inferState.testImageUrl) {
        InferState.testImageUrl = inferState.testImageUrl;
    }
    InferState.testImageWidth = inferState.testImageWidth || 0;
    InferState.testImageHeight = inferState.testImageHeight || 0;
    InferState.currentGalleryFolder = inferState.currentGalleryFolder || '';
    
    // 等待数据加载完成后恢复 UI 选中状态
    setTimeout(() => {
        // 恢复选中的训练任务
        if (InferState.selectedTask) {
            const taskEl = document.querySelector(`.task-select-item[onclick*="${InferState.selectedTask}"]`);
            if (taskEl) {
                taskEl.classList.add('selected');
                // 显示 epoch 选择区并选中 epochs
                const epochSection = document.getElementById('epoch-select-section');
                if (epochSection) epochSection.style.display = 'block';
                
                const task = InferState.trainingTasks?.find(t => t.task_id === InferState.selectedTask);
                if (task) {
                    renderEpochGrid(task.epochs);
                    // 自动填入该任务的触发词
                    const promptInput = document.getElementById('infer-prompt');
                    if (promptInput) {
                        let tw = task.trigger_word || '';
                        if (!tw && task.task_name) {
                            const parts = task.task_name.split('_');
                            if (parts.length >= 3 && /^\d+$/.test(parts[0]) && /^\d+$/.test(parts[1])) {
                                tw = parts.slice(2).join('_');
                            }
                        }
                        if (tw) promptInput.value = tw;
                    }
                }
            }
        }
        
        // 恢复图片预览
        if (InferState.testImages && InferState.testImages.length > 0) {
            const preview = document.getElementById('test-image-preview');
            if (preview) {
                preview.innerHTML = `
                    <div class="test-images-grid">
                        ${InferState.testImages.map(img => 
                            `<img src="${img.url}" alt="" loading="lazy">`
                        ).join('')}
                    </div>
                `;
            }
        }
    }, 500);
}

// 自动保存状态（在关键操作时）
function autoSaveState() {
    // 延迟保存，避免频繁写入
    clearTimeout(window._saveStateTimer);
    window._saveStateTimer = setTimeout(saveAppState, 300);
}

// 监听页面关闭前保存状态并提示未保存
window.addEventListener('beforeunload', (e) => {
    saveAppState();
    // 如果有未保存的新任务更改，提示用户
    if (State.isDirty && State.isNewTask) {
        e.preventDefault();
        e.returnValue = '您有未保存的任务数据，确定要离开吗？';
        return e.returnValue;
    }
});

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
            console.log('[API] createTask 开始提交', data);
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 60秒超时
            
            const res = await fetch('/api/v1/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            console.log('[API] createTask 响应状态:', res.status);
            const result = await res.json();
            console.log('[API] createTask 响应数据:', result);
            return result;
        } catch (e) {
            console.error('[API] createTask error:', e);
            if (e.name === 'AbortError') {
                return { code: 500, message: '请求超时，请检查服务是否正常运行' };
            }
            return { code: 500, message: e.message };
        }
    },
    getRawVideos: (taskId) => API.request(`/preprocess/raw${taskId ? '?task_id=' + taskId : ''}`),
    processVideos: (data) => API.request('/preprocess/videos', { method: 'POST', body: JSON.stringify(data) }),
    getProcessedVideos: (taskId) => API.request(`/preprocess/list${taskId ? '?task_id=' + taskId : ''}`),
    getArBuckets: (taskId) => API.request(`/preprocess/ar-buckets${taskId ? '?task_id=' + taskId : ''}`),
    updateCaption: (data) => API.request('/preprocess/caption', { method: 'PUT', body: JSON.stringify(data) }),
    translateCaption: (data) => API.request('/preprocess/translate', { method: 'POST', body: JSON.stringify(data) }),
    recaptionVideo: (data) => API.request('/preprocess/recaption', { method: 'POST', body: JSON.stringify(data) }),
    batchReplaceTrigger: (data) => API.request('/preprocess/batch-replace-trigger', { method: 'POST', body: JSON.stringify(data) }),
    deleteVideo: (filename, taskId) => API.request(`/preprocess/video/${filename}${taskId ? '?task_id=' + taskId : ''}`, { method: 'DELETE' }),
    deleteRawVideo: (filename, taskId) => API.request(`/preprocess/raw/${filename}${taskId ? '?task_id=' + taskId : ''}`, { method: 'DELETE' }),
    getFrame: (filename, frameNumber, taskId) => `/api/v1/preprocess/frame/${encodeURIComponent(filename)}?frame=${frameNumber}${taskId ? '&task_id=' + taskId : ''}`,
    saveDraft: (data) => API.request('/training/draft', { method: 'POST', body: JSON.stringify(data) }),
    copyTask: (taskId, newName) => API.request('/training/copy', { method: 'POST', body: JSON.stringify({ task_id: taskId, new_name: newName }) }),
    restartTask: (taskId) => API.request(`/training/restart/${taskId}`, { method: 'POST' }),
    resumeTask: (taskId) => API.request(`/training/resume/${taskId}`, { method: 'POST' }),
    getCheckpoints: (taskId) => API.request(`/training/${taskId}/checkpoints`)
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
    lr: '8e-5',
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
async function showHomePage(skipConfirm = false) {
    // 如果正在处理数据，提示用户处理会在后台继续
    if (State.isProcessing) {
        if (!confirm('数据正在处理中，返回主页后处理将在后台继续。是否返回？')) {
            return;
        }
    }
    
    // 如果有未保存的更改，自动保存为草稿
    if (State.isDirty || State.isNewTask) {
        const taskName = document.getElementById('task-name')?.value?.trim();
        if (taskName && State.currentTaskId) {
            try {
                await autoSaveDraftSilently();
            } catch (e) {
                console.warn('自动保存草稿失败:', e);
            }
        }
    }
    
    // 清除新任务状态
    State.isDirty = false;
    State.isNewTask = false;
    State.currentTaskId = null;
    
    document.getElementById('page-home').classList.add('active');
    document.getElementById('page-training').classList.remove('active');
    State.currentPage = 'home';
    localStorage.setItem('currentPage', 'home');
    loadTasks();
    loadGpus();
    autoSaveState();
}

function showTrainingPage() {
    document.getElementById('page-home').classList.remove('active');
    document.getElementById('page-training').classList.add('active');
    State.currentPage = 'training';
    localStorage.setItem('currentPage', 'training');
    loadGpus();
    
    // 如果是新建任务
    if (State.isNewTask) {
        // 新建任务：加载上次配置，设置任务名称，清空视频
        applyConfigToForm(loadLastConfig());
        document.getElementById('task-name').value = generateTaskName();
        // 清空触发词（新任务不继承上次的触发词）
        const twInput = document.getElementById('trigger-word');
        if (twInput) twInput.value = '';
        // 清空视频状态
        State.rawVideos = [];
        State.processedVideos = [];
        updateRawVideoDisplay();
        updateProcessedDisplay();
        // 默认切换到原始视频 tab
        switchDataTab('raw');
        // 隐藏训练状态标签
        updateTrainingStatusBadge(null);
    } else if (State.currentTaskId) {
        // 编辑已有任务：加载任务数据
        loadTaskData(State.currentTaskId);
    } else {
        // 其他情况：加载上次配置
        applyConfigToForm(loadLastConfig());
        document.getElementById('task-name').value = '';
        State.rawVideos = [];
        State.processedVideos = [];
        updateRawVideoDisplay();
        updateProcessedDisplay();
    }
    autoSaveState();
}

function restorePage() {
    // 使用新的状态恢复系统
    restoreAppState();
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
    // 切换到已处理 tab 时重新计算所有 textarea 高度
    if (tab === 'processed') {
        requestAnimationFrame(autoResizeAllTextareas);
    }
    autoSaveState();
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
    State.isNewTask = false;  // 明确标记这不是新任务
    State.isDirty = false;    // 初始化为无修改状态
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
        
        // 恢复触发词
        if (task.trigger_word) {
            const tw = document.getElementById('trigger-word');
            if (tw) tw.value = task.trigger_word;
        }
        
        // AR buckets
        if (task.dataset?.ar_buckets) {
            const arInput = document.getElementById('ar-buckets');
            if (arInput) arInput.value = task.dataset.ar_buckets.join(', ');
        }
        
        renderGpus();
        
        // 更新训练状态标签
        updateTrainingStatusBadge(task.status, task.description);
        
        // 延迟重算 textarea 高度（确保视频/tab 都已渲染）
        setTimeout(autoResizeAllTextareas, 800);
    }
}

// 更新训练数据旁边的训练状态标签
function updateTrainingStatusBadge(status, taskName) {
    const badge = document.getElementById('training-status-badge');
    if (!badge) return;
    
    if (!status || status === 'draft') {
        badge.style.display = 'none';
        return;
    }
    
    const statusMap = { 
        'queued': '排队中', 
        'running': '训练中', 
        'completed': '已完成', 
        'failed': '报错', 
        'stopped': '已停止'
    };
    
    badge.textContent = statusMap[status] || status;
    badge.className = `status-badge-inline ${status}`;
    badge.style.display = 'inline-block';
    badge.title = '点击查看训练状态';
    // 保存当前任务名称用于弹窗
    badge.dataset.taskName = taskName || '';
}

// 从训练数据页的状态标签打开训练状态弹窗
function openTrainingStatusFromBadge() {
    if (State.currentTaskId) {
        const badge = document.getElementById('training-status-badge');
        const taskName = badge?.dataset?.taskName || '';
        showTrainingStatus(State.currentTaskId, taskName);
    }
}
window.openTrainingStatusFromBadge = openTrainingStatusFromBadge;

// 新建任务
function startNewTraining() {
    // 检查是否有未保存的更改
    if (State.isDirty && State.isNewTask) {
        if (!confirm('当前任务有未保存的更改，确定要放弃并新建任务吗？')) {
            return;
        }
    }
    
    console.log('startNewTraining called');
    // 生成新的任务ID
    State.currentTaskId = generateTaskId();
    State.rawVideos = [];
    State.processedVideos = [];
    State.selectedGpu = null;
    State.lossHistory = [];
    State.isDirty = false;
    State.isNewTask = true;
    
    // showTrainingPage 会检测 isNewTask 并正确初始化
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
    const dotsHtml = State.gpus.map(g => {
        const isAvail = g.status === 'available';
        return `<div class="gpu-dot ${g.status}" title="GPU ${g.gpu_id}: ${isAvail ? '空闲' : '使用中'}"></div>`;
    }).join('');
    
    // 渲染到所有 GPU 指示器容器
    ['gpu-indicators', 'home-gpu-indicators', 'infer-gpu-indicators'].forEach(id => {
        const c = document.getElementById(id);
        if (c) c.innerHTML = dotsHtml;
    });
}

// ============================================
// Raw Videos
// ============================================
async function loadRawVideos() {
    const res = await API.getRawVideos(State.currentTaskId);
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
    const taskParam = State.currentTaskId ? `?task_id=${State.currentTaskId}` : '';
    grid.innerHTML = State.rawVideos.map(v => `
        <div class="raw-video-item">
            <video src="/api/v1/preprocess/raw/file/${encodeURIComponent(v.filename)}${taskParam}" 
                   muted loop onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;">
            </video>
            <div class="name">${v.filename}</div>
            <button class="remove-btn" onclick="event.stopPropagation(); removeRawVideo('${v.filename}')">×</button>
        </div>
    `).join('');
}

async function removeRawVideo(filename) {
    if (!confirm(`删除 ${filename}？`)) return;
    const res = await API.deleteRawVideo(filename, State.currentTaskId);
    if (res.code === 200) {
        toast('已删除', 'success');
        markDirty();
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
    
    // 添加 task_id
    if (State.currentTaskId) {
        formData.append('task_id', State.currentTaskId);
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
            markDirty();
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
    const summaryCount = document.getElementById('summary-video-count');
    const videoGrid = document.getElementById('process-video-grid');
    
    if (!modal) return;
    
    // 同步触发词：从主界面输入框读取并填入模态框
    const mainTriggerWord = document.getElementById('trigger-word')?.value || '';
    const modalTriggerWord = document.getElementById('modal-trigger-word');
    if (modalTriggerWord && mainTriggerWord) {
        modalTriggerWord.value = mainTriggerWord;
    }
    
    // 更新视频数量
    if (summaryCount) summaryCount.textContent = State.rawVideos.length;
    
    // 渲染所有视频缩略图
    if (videoGrid) {
        const taskParam = State.currentTaskId ? `?task_id=${State.currentTaskId}` : '';
        videoGrid.innerHTML = State.rawVideos.map(v => `
            <div class="process-video-item">
                <video src="/api/v1/preprocess/raw/file/${encodeURIComponent(v.filename)}${taskParam}" 
                       muted loop onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;">
                </video>
                <div class="video-name">${v.filename}</div>
            </div>
        `).join('');
    }
    
    modal.style.display = 'flex';
}

function hideProcessModal() {
    const modal = document.getElementById('process-modal');
    if (modal) modal.style.display = 'none';
    
    // 同步触发词回主界面
    const modalTriggerWord = document.getElementById('modal-trigger-word')?.value || '';
    const mainTriggerWord = document.getElementById('trigger-word');
    if (mainTriggerWord && modalTriggerWord) {
        mainTriggerWord.value = modalTriggerWord;
    }
}

// 保留函数以兼容旧代码
function loadVideoForPreview() {}
function updateFramePreview() {}

async function confirmProcessing() {
    if (State.isProcessing) return;
    
    const triggerWord = document.getElementById('modal-trigger-word')?.value || '';
    const fps = parseInt(document.getElementById('modal-fps')?.value || 16);
    
    hideProcessModal();
    State.isProcessing = true;
    showProcessStatus('正在处理...');
    
    try {
        const res = await API.processVideos({
            task_id: State.currentTaskId,
            input_dir: CONFIG.rawPath,
            output_dir: CONFIG.datasetPath,
            prompt_prefix: triggerWord,
            fps: fps,
            use_qwen_vl: true
        });
        
        if (res.code === 200 || res.code === 201) {
            hideProcessStatus();
            toast(`处理完成！${res.data?.processed?.length || 0} 个视频`, 'success');
            markDirty();
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
    const res = await API.getProcessedVideos(State.currentTaskId);
    State.processedVideos = res.data?.videos || [];
    updateProcessedDisplay();
    loadArBuckets();
}

async function loadArBuckets() {
    try {
        const res = await API.getArBuckets(State.currentTaskId);
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
        if (list) { list.style.display = 'grid'; renderProcessedVideos(); }
        if (empty) empty.style.display = 'none';
    } else {
        if (list) list.style.display = 'none';
        if (empty) empty.style.display = 'flex';
    }
}

function renderProcessedVideos() {
    const list = document.getElementById('processed-list');
    if (!list) return;
    const taskParam = State.currentTaskId ? `?task_id=${State.currentTaskId}` : '';
    list.innerHTML = State.processedVideos.map((v, i) => `
        <div class="processed-card" data-id="${i}">
            <div class="card-video">
                <video src="/api/v1/preprocess/video/${v.filename}${taskParam}" muted loop onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0;"></video>
                <div class="card-filename">${v.filename}</div>
            </div>
            <div class="card-body">
                <div class="card-caption">
                    <textarea id="caption-${i}" onchange="markChanged(${i})" oninput="autoResizeTextarea(this)" placeholder="输入提示词...">${v.caption || ''}</textarea>
                </div>
                <div class="card-actions">
                    <button class="btn-icon" onclick="saveCaption(${i}, '${v.filename}')" title="保存">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17,21 17,13 7,13 7,21"/><polyline points="7,3 7,8 15,8"/></svg>
                    </button>
                    <button class="btn-icon" id="btn-translate-${i}" onclick="toggleTranslate(${i})" title="翻译">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2v3"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/></svg>
                    </button>
                    <button class="btn-icon" onclick="recaptionVideo(${i}, '${v.filename}')" title="重新打标">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 105.64-11.36L1 10"/></svg>
                    </button>
                    <button class="btn-icon btn-icon-danger" onclick="deleteProcessedVideo('${v.filename}')" title="删除">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="3,6 5,6 21,6"/><path d="M19,6v14a2,2 0 01-2,2H7a2,2 0 01-2-2V6m3,0V4a2,2 0 012-2h4a2,2 0 012,2v2"/></svg>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
    
    // 延迟自适应 textarea 高度
    requestAnimationFrame(autoResizeAllTextareas);
    setTimeout(autoResizeAllTextareas, 500);
}

// 文本框自适应高度
function autoResizeTextarea(el) {
    if (!el || !el.offsetParent) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight + 2, 120) + 'px';
}

// 批量调整
function autoResizeAllTextareas() {
    document.querySelectorAll('.processed-card textarea').forEach(el => {
        autoResizeTextarea(el);
    });
}

function markChanged(id) {
    const item = document.querySelector(`.processed-card[data-id="${id}"]`);
    if (item) item.classList.add('changed');
}

async function saveCaption(id, filename) {
    const textarea = document.getElementById(`caption-${id}`);
    if (!textarea) return;
    try {
        const res = await API.updateCaption({ filename, caption: textarea.value, task_id: State.currentTaskId });
        if (res.code === 200) {
            toast('已保存', 'success');
            markDirty();
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
        const res = await API.deleteVideo(filename, State.currentTaskId);
        if (res.code === 200) {
            toast('已删除', 'success');
            markDirty();
            loadProcessedVideos();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

// 一键翻译（英译中），再次点击返回原文
const _translateCache = {};
async function toggleTranslate(id) {
    const textarea = document.getElementById(`caption-${id}`);
    const btn = document.getElementById(`btn-translate-${id}`);
    if (!textarea) return;

    // 如果当前是翻译态，切换回原文
    if (textarea.dataset.translated === 'true') {
        textarea.value = textarea.dataset.original;
        textarea.readOnly = false;
        textarea.dataset.translated = 'false';
        if (btn) btn.classList.remove('btn-translate-active');
        return;
    }

    // 保存原文
    const original = textarea.value;
    textarea.dataset.original = original;

    // 有缓存就直接用
    if (_translateCache[original]) {
        textarea.value = _translateCache[original];
        textarea.readOnly = true;
        textarea.dataset.translated = 'true';
        if (btn) btn.classList.add('btn-translate-active');
        return;
    }

    textarea.value = '翻译中...';
    textarea.readOnly = true;
    try {
        const res = await API.translateCaption({ text: original });
        if (res.code === 200 && res.data?.translated) {
            _translateCache[original] = res.data.translated;
            textarea.value = res.data.translated;
            textarea.dataset.translated = 'true';
            if (btn) btn.classList.add('btn-translate-active');
        } else {
            textarea.value = original;
            textarea.readOnly = false;
            toast('翻译失败', 'error');
        }
    } catch (e) {
        textarea.value = original;
        textarea.readOnly = false;
        toast(`翻译失败: ${e.message}`, 'error');
    }
}

// 重新打标（QwenVL + 触发词）
async function recaptionVideo(id, filename) {
    const textarea = document.getElementById(`caption-${id}`);
    const translateBtn = document.getElementById(`btn-translate-${id}`);
    if (!textarea) return;
    // 恢复原文编辑态
    textarea.readOnly = false;
    textarea.dataset.translated = 'false';
    if (translateBtn) translateBtn.classList.remove('btn-translate-active');

    const triggerWord = document.getElementById('trigger-word')?.value || '';
    textarea.value = '正在重新打标...';
    textarea.classList.add('recaptioning');
    try {
        const res = await API.recaptionVideo({ filename, trigger_word: triggerWord, task_id: State.currentTaskId });
        textarea.classList.remove('recaptioning');
        if (res.code === 200 && res.data?.caption) {
            textarea.value = res.data.caption;
            // 同步更新 State
            if (State.processedVideos[id]) {
                State.processedVideos[id].caption = res.data.caption;
            }
            markDirty();
            toast('打标完成', 'success');
        } else {
            throw new Error(res.message || '打标失败');
        }
    } catch (e) {
        textarea.classList.remove('recaptioning');
        textarea.value = '';
        toast(`打标失败: ${e.message}`, 'error');
    }
}

// 批量修改触发词
async function modifyTriggerWord() {
    const input = document.getElementById('trigger-word');
    if (!input) return;
    const newTrigger = input.value.trim();
    if (!newTrigger) {
        toast('请先输入触发词', 'warning');
        return;
    }

    // 从第一个已处理视频的 caption 中推测旧触发词
    let oldTrigger = '';
    if (State.processedVideos.length > 0) {
        const firstCaption = State.processedVideos[0].caption || '';
        const commaIdx = firstCaption.indexOf(',');
        if (commaIdx > 0) {
            oldTrigger = firstCaption.substring(0, commaIdx).trim();
        }
    }

    if (!confirm(`将所有已处理视频的触发词从「${oldTrigger || '(无)'}」修改为「${newTrigger}」？`)) return;

    try {
        const res = await API.batchReplaceTrigger({ old_trigger: oldTrigger, new_trigger: newTrigger, task_id: State.currentTaskId });
        if (res.code === 200) {
            toast(`已更新 ${res.data.updated} 个文件`, 'success');
            markDirty();
            loadProcessedVideos();
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`修改失败: ${e.message}`, 'error');
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
    console.log('[startTraining] 函数被调用');
    console.log('[startTraining] State.processedVideos:', State.processedVideos);
    console.log('[startTraining] State.processedVideos.length:', State.processedVideos.length);
    
    if (State.processedVideos.length === 0) {
        console.log('[startTraining] 没有处理后的视频，显示警告');
        toast('请先处理训练数据', 'warning');
        switchDataTab('processed');
        return;
    }
    let name = document.getElementById('task-name')?.value?.trim();
    console.log('[startTraining] 任务名称:', name);
    if (!name) {
        console.log('[startTraining] 任务名称为空，显示警告');
        toast('请输入任务名称', 'warning');
        document.getElementById('task-name')?.focus();
        return;
    }
    
    const btn = document.getElementById('start-btn');
    const resetBtn = () => {
        console.log('[startTraining] 重置按钮状态');
        if (btn) { btn.disabled = false; btn.textContent = '提交训练'; }
    };
    
    if (btn) { btn.disabled = true; btn.textContent = '提交中...'; }
    
    console.log('[startTraining] 开始提交训练任务...');
    
    try {
        const formValues = getFormValues();
        
        // 保存当前配置供下次使用
        saveLastConfig(getCurrentConfigFromForm());
        
        const taskData = {
            task_id: State.currentTaskId,
            gpu_id: -1,  // -1 表示自动分配GPU
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
        };
        
        console.log('[startTraining] 任务数据:', taskData);
        
        const res = await API.createTask(taskData);
        
        console.log('[startTraining] API 响应:', res);
        
        if (res.code === 201 || res.code === 200) {
            const taskId = res.data?.task_id;
            const status = res.data?.status;
            if (status === 'queued') {
                toast('任务已加入排队队列！', 'success');
            } else {
                toast('训练任务已提交！', 'success');
            }
            clearDirty();
            State.isNewTask = false;
            State.lossHistory = [];
            resetBtn();
            if (taskId) showTrainingStatus(taskId, name);
            else setTimeout(() => showHomePage(), 1500);
        } else {
            resetBtn();
            throw new Error(res.message || '提交失败');
        }
    } catch (e) {
        console.error('[startTraining] 错误:', e);
        toast(`提交失败: ${e.message}`, 'error');
        resetBtn();
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
    
    try {
        const formValues = getFormValues();
        saveLastConfig(getCurrentConfigFromForm());
        
        const draftData = {
            task_id: State.currentTaskId,
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
            processed_videos: State.processedVideos.map(v => ({ filename: v.filename, caption: v.caption })),
            trigger_word: document.getElementById('trigger-word')?.value?.trim() || ''
        };
        
        const res = await API.saveDraft(draftData);
        if (res.code === 201 || res.code === 200) {
            toast('草稿保存成功！', 'success');
            clearDirty();
            State.isNewTask = false;
            // 更新 task_id 为服务器返回的正式 ID
            if (res.data?.task_id) {
                State.currentTaskId = res.data.task_id;
            }
            loadTasks();
        } else {
            throw new Error(res.message || '保存失败');
        }
    } catch (e) {
        toast(`保存失败: ${e.message}`, 'error');
    }
}

// 静默自动保存草稿（返回主页时调用，不弹 toast）
async function autoSaveDraftSilently() {
    let name = document.getElementById('task-name')?.value?.trim();
    if (!name) name = generateTaskName();
    
    const formValues = getFormValues();
    const draftData = {
        task_id: State.currentTaskId,
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
        processed_videos: State.processedVideos.map(v => ({ filename: v.filename, caption: v.caption })),
        trigger_word: document.getElementById('trigger-word')?.value?.trim() || ''
    };
    
    const res = await API.saveDraft(draftData);
    if (res.code === 201 || res.code === 200) {
        console.log('[autoSave] 草稿已自动保存');
        if (res.data?.task_id) {
            State.currentTaskId = res.data.task_id;
        }
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
                        <div class="card-value"><span id="status-gpu">-</span> <span id="status-memory" style="font-size:11px;color:var(--text-muted);"></span></div>
                    </div>
                    <div class="status-card">
                        <div class="card-label">Epoch</div>
                        <div class="card-value" id="status-epoch">0 / -</div>
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
                
                <div class="error-message-section" id="error-message-section" style="display:none;">
                    <div style="padding: 10px 14px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; color: #f87171; font-size: 13px; line-height: 1.5;">
                        <span style="font-weight: 600;">错误信息: </span><span id="error-message-text"></span>
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

                <div class="epoch-models-section" id="epoch-models-section" style="display:none;">
                    <div class="chart-header">
                        <h4>LoRA 模型</h4>
                        <span class="epoch-count" id="epoch-model-count">0 个</span>
                    </div>
                    <div class="epoch-model-list" id="epoch-model-list">
                        <div class="log-placeholder">加载中...</div>
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
    
    // 加载 epoch 模型列表
    try {
        const epochsRes = await API.request(`/training/${taskId}/epochs`);
        if (epochsRes.code === 200 && epochsRes.data) {
            updateEpochModelList(epochsRes.data.epochs || [], taskId);
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
        const usedGB = (mem.used / 1024).toFixed(1);
        const totalGB = (mem.total / 1024).toFixed(1);
        memoryEl.textContent = `(${usedGB}/${totalGB} GB)`;
    }
    
    // Loss 处理 —— 优先使用后端返回的完整 step_losses
    if (data.metrics?.step_losses?.length > 0) {
        const stepLosses = data.metrics.step_losses;
        const latestLoss = stepLosses[stepLosses.length - 1];
        if (lossValueEl) lossValueEl.textContent = `当前: ${latestLoss.toFixed(4)}`;
        
        // 用完整的 step_losses 替换前端缓存
        State.lossHistory = stepLosses.map((loss, i) => ({ step: i + 1, loss }));
        drawLossChart();
    } else if (data.metrics?.current_loss != null) {
        const loss = data.metrics.current_loss;
        if (lossValueEl) lossValueEl.textContent = `当前: ${loss.toFixed(4)}`;
    }
    
    // 显示/隐藏错误信息
    const errorSection = document.getElementById('error-message-section');
    const errorText = document.getElementById('error-message-text');
    if (errorSection && errorText) {
        if (data.error_message && ['failed', 'stopped'].includes(data.status)) {
            errorText.textContent = data.error_message;
            errorSection.style.display = 'block';
        } else {
            errorSection.style.display = 'none';
        }
    }

    if (['completed', 'failed', 'stopped'].includes(data.status)) {
        // 只在状态轮询中检测到完成时显示 toast（避免打开已完成任务的弹窗时重复提示）
        if (statusPollInterval) {
            if (data.status === 'completed') toast('训练完成！', 'success');
            else if (data.status === 'failed') toast(`训练失败: ${data.error_message || '未知错误'}`, 'error');
        }
        stopStatusPolling();
    }
    
    // 更新footer按钮
    const footer = document.querySelector('.status-footer');
    if (footer && data.task_id) {
        if (['failed', 'stopped'].includes(data.status)) {
            footer.innerHTML = `
                <button class="btn-primary" onclick="resumeTraining('${data.task_id}')">断点续训</button>
                <button class="btn-secondary" onclick="restartTraining('${data.task_id}')">重新训练</button>
                <button class="btn-secondary" onclick="hideTrainingStatus()">关闭</button>
            `;
        } else if (data.status === 'completed') {
            footer.innerHTML = `
                <button class="btn-primary" onclick="resumeTraining('${data.task_id}')">断点续训</button>
                <button class="btn-secondary" onclick="hideTrainingStatus()">关闭</button>
            `;
        } else if (['running', 'queued'].includes(data.status)) {
            footer.innerHTML = `
                <button class="btn-danger" onclick="stopTraining('${data.task_id}')">停止训练</button>
                <button class="btn-secondary" onclick="hideTrainingStatus()">关闭</button>
            `;
        } else {
            footer.innerHTML = `
                <button class="btn-secondary" onclick="hideTrainingStatus()">关闭</button>
            `;
        }
    }
}

function updateEpochModelList(epochs, taskId) {
    const section = document.getElementById('epoch-models-section');
    const container = document.getElementById('epoch-model-list');
    const countEl = document.getElementById('epoch-model-count');
    
    if (!section || !container) return;
    
    if (!epochs || epochs.length === 0) {
        section.style.display = 'none';
        return;
    }
    
    section.style.display = 'block';
    if (countEl) countEl.textContent = `${epochs.length} 个`;
    
    container.innerHTML = epochs.map(ep => {
        const sizeMB = ep.size_mb ? ep.size_mb.toFixed(1) : '?';
        return `
            <div class="epoch-model-item">
                <div class="epoch-model-info">
                    <span class="epoch-model-name">epoch${ep.epoch}</span>
                    <span class="epoch-model-size">${sizeMB} MB</span>
                </div>
                <a class="btn-epoch-download" href="/api/v1/training/${taskId}/epoch/${ep.epoch}/download" title="下载 LoRA 模型">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                </a>
            </div>`;
    }).join('');
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

async function restartTraining(taskId) {
    if (!confirm('确定要重新提交训练吗？将清除旧日志并重新开始。')) return;
    
    try {
        const res = await API.restartTask(taskId);
        if (res.code === 200) {
            const queuePos = res.data?.queue_position;
            if (queuePos && queuePos > 0) {
                toast(`训练任务已重新加入队列，位置: ${queuePos}`, 'success');
            } else {
                toast('训练任务已重新提交', 'success');
            }
            // 重新开始状态轮询
            showTrainingStatus(taskId);
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`重新提交失败: ${e.message}`, 'error');
    }
}
window.restartTraining = restartTraining;

async function resumeTraining(taskId) {
    if (!confirm('确定要从最近的 checkpoint 断点续训吗？')) return;
    
    try {
        const res = await API.resumeTask(taskId);
        if (res.code === 200) {
            toast('断点续训已启动', 'success');
            // 清除 config 中的 resume 标记（仅影响本次启动）
            showTrainingStatus(taskId);
        } else {
            throw new Error(res.message);
        }
    } catch (e) {
        toast(`断点续训失败: ${e.message}`, 'error');
    }
}
window.resumeTraining = resumeTraining;

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
    
    State.currentPage = page;
    localStorage.setItem('currentPage', page);
    autoSaveState();
    
    // 所有页面都加载 GPU 状态（用于顶栏指示器）
    loadGpus();
    
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

// 训练任务筛选状态
let taskFilterStatus = 'all';
let taskSearchKeyword = '';

async function loadTrainingTasksWithLoras() {
    const container = document.getElementById('task-select-list');
    if (!container) return;
    
    container.innerHTML = '<div class="empty-state"><p>加载中...</p></div>';
    
    try {
        const res = await API.request('/inference/tasks-with-loras');
        if (res.code === 200 && res.data?.tasks) {
            InferState.trainingTasks = res.data.tasks;
            renderTrainingTaskList();
        } else {
            throw new Error(res.message || '获取失败');
        }
    } catch (e) {
        container.innerHTML = `<div class="empty-state error"><p>加载失败: ${e.message}</p></div>`;
    }
}

function renderTrainingTaskList() {
    const container = document.getElementById('task-select-list');
    if (!container) return;
    
    let tasks = InferState.trainingTasks || [];
    
    // 搜索过滤
    if (taskSearchKeyword) {
        const kw = taskSearchKeyword.toLowerCase();
        tasks = tasks.filter(t => 
            (t.task_name || '').toLowerCase().includes(kw) || 
            (t.task_id || '').toLowerCase().includes(kw)
        );
    }
    
    // 状态过滤
    if (taskFilterStatus !== 'all') {
        tasks = tasks.filter(t => t.task_status === taskFilterStatus);
    }
    
    if (tasks.length === 0) {
        const msg = taskSearchKeyword || taskFilterStatus !== 'all' 
            ? '没有匹配的任务' 
            : '暂无训练任务';
        container.innerHTML = `<div class="empty-state"><p>${msg}</p></div>`;
        return;
    }
    
    container.innerHTML = tasks.map(task => {
        const readyCount = task.epochs.filter(e => e.ready).length;
        const pendingCount = task.epochs.filter(e => e.pending).length;
        const statusBadge = getTaskStatusBadge(task.task_status);
        const metaText = pendingCount > 0 
            ? `${readyCount} 已就绪 · ${pendingCount} 训练中`
            : `${task.epochs.length} 个版本`;
        const timeText = task.created_at ? formatDate(task.created_at) : '';
        return `
        <div class="task-select-item ${InferState.selectedTask === task.task_id ? 'selected' : ''}" 
             onclick="selectTrainingTask('${task.task_id}')"
             data-task-id="${task.task_id}"
             data-task-status="${task.task_status}">
            <div class="task-select-info">
                <div class="task-select-header">
                    <span class="task-select-name">${task.task_name || task.task_id.slice(-12)}</span>
                    ${statusBadge}
                </div>
                <span class="task-select-meta">${metaText}${timeText ? ' · ' + timeText : ''}</span>
            </div>
        </div>
    `}).join('');
}

function filterTrainingTasks() {
    taskSearchKeyword = document.getElementById('task-search-input')?.value?.trim() || '';
    renderTrainingTaskList();
}

function filterTrainingTasksByStatus(status) {
    taskFilterStatus = status;
    // 更新 tab 激活状态
    document.querySelectorAll('.task-filter-tab').forEach(el => {
        el.classList.toggle('active', el.dataset.filter === status);
    });
    renderTrainingTaskList();
}

function getTaskStatusBadge(status) {
    switch (status) {
        case 'running':
            return '<span class="task-status-badge running">训练中</span>';
        case 'queued':
            return '<span class="task-status-badge queued">排队中</span>';
        case 'completed':
            return '<span class="task-status-badge completed">已完成</span>';
        default:
            return '';
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
        
        // 自动填入该训练任务的触发词
        const promptInput = document.getElementById('infer-prompt');
        if (promptInput) {
            let tw = task.trigger_word || '';
            // 回退：从任务名 MMDD_HHMM_word 中提取
            if (!tw && task.task_name) {
                const parts = task.task_name.split('_');
                if (parts.length >= 3 && /^\d+$/.test(parts[0]) && /^\d+$/.test(parts[1])) {
                    tw = parts.slice(2).join('_');
                }
            }
            if (tw) {
                promptInput.value = tw;
            }
        }
    }
    
    autoSaveState();
}

function renderEpochGrid(epochs) {
    const grid = document.getElementById('epoch-grid');
    if (!grid) return;
    
    grid.innerHTML = epochs.map(ep => {
        const isPending = ep.pending && !ep.ready;
        const isSelected = InferState.selectedLoras.some(l => l.epoch === ep.epoch);
        const sizeText = isPending ? '训练中' : `${ep.size_mb.toFixed(0)}MB`;
        return `
        <div class="epoch-item ${isSelected ? 'selected' : ''} ${isPending ? 'pending' : ''}"
             onclick="toggleEpochSelection('${ep.path || ''}', 'epoch${ep.epoch}', ${ep.epoch}, ${isPending})">
            <span class="epoch-name">Epoch ${ep.epoch}</span>
            <span class="epoch-size">${sizeText}</span>
            ${isPending ? '<span class="pending-indicator"></span>' : ''}
        </div>
    `}).join('');
    
    updateSelectedLorasTags();
}

function toggleEpochSelection(path, name, epoch, isPending = false) {
    const idx = InferState.selectedLoras.findIndex(l => l.epoch === epoch);
    if (idx >= 0) {
        InferState.selectedLoras.splice(idx, 1);
    } else {
        InferState.selectedLoras.push({ path: path || null, name, epoch, pending: isPending });
    }
    
    // 更新 UI
    const task = InferState.trainingTasks.find(t => t.task_id === InferState.selectedTask);
    if (task) {
        renderEpochGrid(task.epochs);
    }
    
    autoSaveState();
}

function updateSelectedLorasTags() {
    // 不再显示已选择标签，直接隐藏容器
    const container = document.getElementById('selected-loras');
    if (container) {
        container.style.display = 'none';
    }
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
            
            // 使用缩略图预加载，瀑布流布局
            grid.innerHTML = res.data.images.map(img => {
                const isSelected = InferState.selectedModalImages.some(s => s.id === img.id);
                const ratio = img.width && img.height ? img.height / img.width : 1;
                return `
                <div class="gallery-image-item waterfall-item ${isSelected ? 'selected' : ''}"
                     style="--aspect-ratio: ${ratio};"
                     data-id="${img.id}"
                     onclick="selectModalImage('${img.id}', '${img.path}', '${img.url}', event)">
                    <img src="${img.thumb_url}" alt="" loading="lazy" onerror="this.src='${img.url}'">
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
    
    // 更新预览 - 显示所有图片，原比例自适应
    const preview = document.getElementById('test-image-preview');
    if (preview) {
        preview.innerHTML = `
            <div class="test-images-grid">
                ${InferState.selectedModalImages.map(img => 
                    `<img src="${img.url}" alt="" loading="lazy">`
                ).join('')}
            </div>
        `;
    }
    
    closeGalleryModal();
    toast(`已选择 ${InferState.selectedModalImages.length} 张图片`, 'success');
    autoSaveState();
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
    
    const triggerWord = document.getElementById('infer-prompt')?.value?.trim();
    if (!triggerWord) {
        toast('请输入触发词', 'warning');
        return;
    }
    
    // 检查是否有选中的图片
    const hasMultipleImages = InferState.testImages && InferState.testImages.length > 1;
    const hasSingleImage = InferState.testImagePath || (InferState.testImages && InferState.testImages.length === 1);
    
    if (!hasSingleImage && !hasMultipleImages) {
        toast('请选择测试图片', 'warning');
        return;
    }
    
    // 获取图片列表
    const imagePaths = hasMultipleImages 
        ? InferState.testImages.map(img => img.path)
        : [InferState.testImages?.[0]?.path || InferState.testImagePath];
    
    // 区分已就绪和pending的epoch
    const readyLoras = InferState.selectedLoras.filter(l => !l.pending && l.path);
    const pendingLoras = InferState.selectedLoras.filter(l => l.pending || !l.path);
    
    // 获取训练任务名称
    const selectedTaskData = InferState.trainingTasks.find(t => t.task_id === InferState.selectedTask);
    const trainingTaskName = selectedTaskData?.task_name || InferState.selectedTask || '';
    
    // 基础参数
    const baseParams = {
        trigger_word: triggerWord,
        lora_strength: parseFloat(document.getElementById('infer-lora-strength')?.value) || 1.0,
        width: parseInt(document.getElementById('infer-width')?.value) || 832,
        height: parseInt(document.getElementById('infer-height')?.value) || 480,
        num_frames: parseInt(document.getElementById('infer-frames')?.value) || 81,
        num_steps: parseInt(document.getElementById('infer-steps')?.value) || 4,
        guidance_scale: parseFloat(document.getElementById('infer-guidance')?.value) || 1.0,
        seed: parseInt(document.getElementById('infer-seed')?.value) || -1,
        use_auto_caption: document.getElementById('infer-auto-caption')?.checked ?? true,
        training_task_id: InferState.selectedTask,  // 训练任务ID
        training_task_name: trainingTaskName        // 训练任务名称
    };
    
    // 生成批次ID
    const isMultiTask = InferState.selectedLoras.length > 1 || imagePaths.length > 1;
    const batchId = isMultiTask ? `batch_${Date.now()}_${Math.random().toString(36).substr(2, 6)}` : null;
    
    // 总任务数（包括预约的）
    const totalTasks = InferState.selectedLoras.length * imagePaths.length;
    const pendingTaskCount = pendingLoras.length * imagePaths.length;
    
    if (pendingTaskCount > 0) {
        toast(`正在创建 ${totalTasks} 个推理任务（${pendingTaskCount} 个为预约任务）...`, 'info');
    } else {
        toast(`正在创建 ${totalTasks} 个推理任务...`, 'info');
    }
    
    try {
        const allTasks = [];
        const scheduledTasks = [];
        
        // 为已就绪的 epoch 创建立即执行的任务
        for (let epochIdx = 0; epochIdx < readyLoras.length; epochIdx++) {
            const lora = readyLoras[epochIdx];
            
            for (let imgIdx = 0; imgIdx < imagePaths.length; imgIdx++) {
                const data = {
                    ...baseParams,
                    lora_path: lora.path,
                    lora_name: lora.name,
                    image_path: imagePaths[imgIdx],
                    batch_id: batchId,
                    epoch_index: epochIdx,
                    image_index: imgIdx
                };
                
                const res = await API.request('/inference/create', {
                    method: 'POST',
                    body: JSON.stringify(data)
                });
                
                if (res.code === 201 && res.data) {
                    allTasks.push(res.data);
                }
            }
        }
        
        // 为pending的 epoch 创建预约任务
        if (pendingLoras.length > 0) {
            const pendingData = {
                training_task_id: InferState.selectedTask,
                epochs: pendingLoras.map(l => l.epoch),
                trigger_word: baseParams.trigger_word,
                image_paths: imagePaths,
                lora_strength: baseParams.lora_strength,
                batch_id: batchId
            };
            
            const pendingRes = await API.request('/inference/create-pending', {
                method: 'POST',
                body: JSON.stringify(pendingData)
            });
            
            if (pendingRes.code === 201 && pendingRes.data) {
                scheduledTasks.push(pendingRes.data);
                // 如果有立即创建的任务，将其加入轮询
                if (pendingRes.data.created_tasks && pendingRes.data.created_tasks.length > 0) {
                    pollBatchInferenceProgress(pendingRes.data.created_tasks.map(t => t.task_id));
                }
                // 启动pending任务的轮询检查（如果有pending任务）
                if (pendingRes.data.pending_tasks && pendingRes.data.pending_tasks.length > 0) {
                    startScheduledInferencePolling(pendingRes.data.batch_id);
                }
            }
        }
        
        if (allTasks.length > 0 || scheduledTasks.length > 0) {
            if (scheduledTasks.length > 0) {
                toast(`已创建 ${allTasks.length} 个立即执行 + ${pendingTaskCount} 个预约推理任务`, 'success');
            } else {
                toast(`已创建 ${allTasks.length} 个推理任务，GPU 4-7 自动分配执行`, 'success');
            }
            
            // 保存当前批次信息用于网格展示
            InferState.currentBatch = {
                batchId,
                epochs: InferState.selectedLoras.map(l => l.name),
                imageCount: imagePaths.length,
                tasks: allTasks,
                scheduledTasks: scheduledTasks,
                visibleEpochs: new Set(InferState.selectedLoras.map((_, i) => i))
            };
            
            // 开始轮询已创建的任务
            if (allTasks.length > 0) {
                pollBatchInferenceProgress(allTasks.map(t => t.task_id));
            }
            
            // 显示网格结果区
            updateBatchResultArea();
            
            loadInferenceHistory();
        } else {
            throw new Error('没有成功创建任何任务');
        }
    } catch (e) {
        toast(`创建失败: ${e.message}`, 'error');
    }
}

// 预约推理轮询（每5分钟检查一次）
function startScheduledInferencePolling(batchId) {
    const POLL_INTERVAL = 5 * 60 * 1000; // 5分钟
    let pollCount = 0;
    const MAX_POLLS = 24 * 12; // 最多轮询24小时（每5分钟一次）
    
    const checkSchedule = async () => {
        pollCount++;
        if (pollCount > MAX_POLLS) {
            console.log('预约推理轮询已达最大次数，停止轮询');
            return;
        }
        
        try {
            // 重新获取任务列表，检查是否有新完成的任务
            const res = await API.request('/inference/tasks-with-loras');
            if (res.code === 200 && res.data?.tasks) {
                // 检查当前选中任务的epoch状态
                const selectedTask = res.data.tasks.find(t => t.task_id === InferState.selectedTask);
                if (selectedTask) {
                    // 找出新就绪的epoch
                    const newlyReady = selectedTask.epochs.filter(ep => 
                        ep.ready && InferState.selectedLoras.some(l => l.epoch === ep.epoch && l.pending)
                    );
                    
                    if (newlyReady.length > 0) {
                        toast(`${newlyReady.length} 个新的 LoRA 已就绪，正在创建推理任务...`, 'info');
                        // 刷新任务列表
                        await loadTrainingTasksWithLoras();
                        loadInferenceHistory();
                    }
                    
                    // 检查是否还有pending的epoch
                    const stillPending = selectedTask.epochs.filter(ep => ep.pending && !ep.ready);
                    if (stillPending.length === 0) {
                        toast('所有预约的 LoRA 已就绪！', 'success');
                        return; // 停止轮询
                    }
                }
                
                // 继续轮询
                setTimeout(checkSchedule, POLL_INTERVAL);
            }
        } catch (e) {
            console.error('检查预约推理状态失败:', e);
            // 出错也继续轮询
            setTimeout(checkSchedule, POLL_INTERVAL);
        }
    };
    
    // 首次检查延迟5分钟
    setTimeout(checkSchedule, POLL_INTERVAL);
    toast('预约推理任务已创建，系统将每5分钟检查一次训练进度', 'info');
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
            <div class="result-success single-video">
                <video controls autoplay loop>
                    <source src="/api/v1/inference/output/${task.task_id}/video" type="video/mp4">
                </video>
                <div class="result-info">
                    <span>${task.lora_name || ''}</span>
                    <a href="/api/v1/inference/output/${task.task_id}/video" download class="btn-secondary">下载</a>
                </div>
            </div>
        `;
    } else if (status === 'failed') {
        area.innerHTML = `
            <div class="inference-error">
                <p>推理失败</p>
                <p class="error-msg">${task.error_message || '未知错误'}</p>
                <button class="btn-primary" onclick="retryInferenceTask('${task.task_id}')" style="margin-top: 16px;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                    </svg>
                    重新推理
                </button>
            </div>
        `;
    }
}

// 批量任务轮询
async function pollBatchInferenceProgress(taskIds) {
    const poll = async () => {
        try {
            let allCompleted = true;
            let anyRunning = false;
            
            // 获取所有任务状态
            for (const taskId of taskIds) {
                const res = await API.request(`/inference/task/${taskId}`);
                if (res.code === 200 && res.data) {
                    const task = res.data;
                    // 更新 InferState.currentBatch 中对应任务
                    if (InferState.currentBatch) {
                        const idx = InferState.currentBatch.tasks.findIndex(t => t.task_id === taskId);
                        if (idx >= 0) {
                            InferState.currentBatch.tasks[idx] = task;
                        }
                    }
                    
                    if (['queued', 'loading', 'running'].includes(task.status)) {
                        allCompleted = false;
                        anyRunning = true;
                    }
                }
            }
            
            // 更新网格显示
            updateBatchResultArea();
            
            if (!allCompleted) {
                setTimeout(poll, 2000);
            } else {
                toast('所有推理任务已完成！', 'success');
                loadInferenceHistory();
            }
        } catch (e) {
            console.error('批量轮询失败:', e);
        }
    };
    
    poll();
}

// 网格结果展示（横轴 epoch，纵轴图片）
function updateBatchResultArea() {
    const area = document.getElementById('result-video-area');
    if (!area || !InferState.currentBatch) return;
    
    const batch = InferState.currentBatch;
    const epochs = batch.epochs;
    const imageCount = batch.imageCount;
    const tasks = batch.tasks;
    const visibleEpochs = batch.visibleEpochs;
    
    // 构建任务矩阵 [imageIdx][epochIdx]
    const taskMatrix = [];
    for (let imgIdx = 0; imgIdx < imageCount; imgIdx++) {
        taskMatrix[imgIdx] = [];
        for (let epochIdx = 0; epochIdx < epochs.length; epochIdx++) {
            const task = tasks.find(t => 
                t.image_index === imgIdx && 
                (t.epoch_index === epochIdx || t.lora_name === epochs[epochIdx])
            );
            taskMatrix[imgIdx][epochIdx] = task;
        }
    }
    
    // 构建 epoch 过滤器
    const epochFilterHtml = epochs.map((name, idx) => `
        <label class="epoch-filter-item ${visibleEpochs.has(idx) ? 'active' : ''}">
            <input type="checkbox" ${visibleEpochs.has(idx) ? 'checked' : ''} 
                   onchange="toggleEpochVisibility(${idx})">
            <span>${name}</span>
        </label>
    `).join('');
    
    // 计算可见列数
    const visibleCount = visibleEpochs.size;
    
    // 构建表头（epoch）
    const headerHtml = epochs.map((name, idx) => 
        visibleEpochs.has(idx) ? `<div class="grid-header-cell">${name}</div>` : ''
    ).join('');
    
    // 构建网格内容
    let gridContentHtml = '';
    for (let imgIdx = 0; imgIdx < imageCount; imgIdx++) {
        for (let epochIdx = 0; epochIdx < epochs.length; epochIdx++) {
            if (!visibleEpochs.has(epochIdx)) continue;
            
            const task = taskMatrix[imgIdx][epochIdx];
            gridContentHtml += renderGridCell(task, imgIdx, epochIdx);
        }
    }
    
    area.innerHTML = `
        <div class="batch-result-container">
            <div class="batch-result-wrapper">
                <div class="epoch-filter">
                    <span class="filter-label">Epoch 筛选:</span>
                    ${epochFilterHtml}
                </div>
                <div class="result-grid" style="--epoch-count: ${visibleCount}">
                    <div class="grid-header">
                        ${headerHtml}
                    </div>
                    <div class="grid-body">
                        ${Array.from({length: imageCount}, (_, imgIdx) => `
                            <div class="grid-row">
                                ${epochs.map((_, epochIdx) => 
                                    visibleEpochs.has(epochIdx) ? renderGridCell(taskMatrix[imgIdx][epochIdx], imgIdx, epochIdx) : ''
                                ).join('')}
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 渲染单个网格单元
function renderGridCell(task, imgIdx, epochIdx) {
    if (!task) {
        return `<div class="grid-cell empty"><span>-</span></div>`;
    }
    
    const status = task.status;
    
    if (status === 'completed') {
        return `
            <div class="grid-cell completed">
                <video controls loop muted playsinline>
                    <source src="/api/v1/inference/output/${task.task_id}/video" type="video/mp4">
                </video>
                <a class="cell-download" href="/api/v1/inference/output/${task.task_id}/video" download title="下载">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                    </svg>
                </a>
            </div>`;
    } else if (status === 'failed') {
        return `
            <div class="grid-cell failed" title="${task.error_message || '推理失败'}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                    <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
            </div>`;
    } else {
        const progress = task.progress || 0;
        return `
            <div class="grid-cell running">
                <div class="cell-progress">
                    <div class="cell-progress-bar" style="width: ${progress}%"></div>
                </div>
                <span class="cell-progress-text">${progress.toFixed(0)}%</span>
            </div>`;
    }
}

// 切换 epoch 可见性
function toggleEpochVisibility(epochIdx) {
    if (!InferState.currentBatch) return;
    
    const visibleEpochs = InferState.currentBatch.visibleEpochs;
    if (visibleEpochs.has(epochIdx)) {
        // 至少保留一个可见
        if (visibleEpochs.size > 1) {
            visibleEpochs.delete(epochIdx);
        }
    } else {
        visibleEpochs.add(epochIdx);
    }
    
    updateBatchResultArea();
}

// 推理历史筛选状态
let historyFilterStatus = 'all';
let historySearchKeyword = '';
let historyDisplayCount = 20; // 每次显示的数量
let allHistoryItems = []; // 缓存所有历史条目（处理后的统一格式）

async function loadInferenceHistory() {
    const container = document.getElementById('inference-history-list');
    if (!container) return;
    
    try {
        const res = await API.request('/inference/tasks?limit=200');
        if (res.code === 200 && res.data?.tasks) {
            const tasks = res.data.tasks;
            if (tasks.length === 0) {
                container.innerHTML = '<div class="empty-state small"><p>暂无历史记录</p></div>';
                updateHistoryLoadMore(0, 0);
                return;
            }
            
            // 将所有任务处理为统一的历史条目
            allHistoryItems = buildHistoryItems(tasks);
            historyDisplayCount = 20;
            renderInferenceHistory();
        }
    } catch (e) {
        console.error('加载历史失败:', e);
    }
}

function buildHistoryItems(tasks) {
    // 按 batch_id 分组
    const batches = {};
    const singleTasks = [];
    
    tasks.forEach(task => {
        if (task.batch_id) {
            if (!batches[task.batch_id]) {
                batches[task.batch_id] = [];
            }
            batches[task.batch_id].push(task);
        } else {
            singleTasks.push(task);
        }
    });
    
    const items = [];
    
    // 批量任务条目
    for (const [batchId, batchTasks] of Object.entries(batches)) {
        const completedCount = batchTasks.filter(t => t.status === 'completed').length;
        const runningCount = batchTasks.filter(t => ['pending', 'queued', 'loading', 'running'].includes(t.status)).length;
        const failedCount = batchTasks.filter(t => t.status === 'failed').length;
        const totalCount = batchTasks.length;
        const epochs = [...new Set(batchTasks.map(t => t.lora_name))];
        const imageCount = Math.max(...batchTasks.map(t => t.image_index || 0)) + 1;
        const status = runningCount > 0 ? 'running' : (completedCount === totalCount ? 'completed' : (failedCount > 0 ? 'failed' : 'queued'));
        const firstTask = batchTasks[0];
        
        // 命名：训练任务名称 + 时间
        const taskName = firstTask.training_task_name || extractTaskNameFromPath(firstTask.lora_path) || '未知任务';
        const timeStr = formatDateTime(firstTask.created_at);
        const displayName = `${taskName} · ${timeStr}`;
        
        items.push({
            type: 'batch',
            batchId,
            taskId: firstTask.task_id,
            displayName,
            taskName,
            detail: `${epochs.length} Epoch × ${imageCount} 图片`,
            status,
            statusText: `${completedCount}/${totalCount} 完成`,
            isRunning: runningCount > 0,
            createdAt: firstTask.created_at,
            timestamp: new Date(firstTask.created_at).getTime()
        });
    }
    
    // 单个任务条目
    for (const task of singleTasks) {
        const taskName = task.training_task_name || extractTaskNameFromPath(task.lora_path) || '未知任务';
        const timeStr = formatDateTime(task.created_at);
        const displayName = `${taskName} · ${timeStr}`;
        const isRunning = ['pending', 'queued', 'loading', 'running'].includes(task.status);
        
        items.push({
            type: 'single',
            taskId: task.task_id,
            displayName,
            taskName,
            detail: `${task.lora_name || ''} · ${(task.trigger_word || task.prompt || '').substring(0, 20)}`,
            status: task.status,
            statusText: getInferenceStatusText(task.status) + (task.status === 'pending' ? '' : isRunning ? ` (${task.progress?.toFixed(0) || 0}%)` : ''),
            isRunning,
            createdAt: task.created_at,
            timestamp: new Date(task.created_at).getTime()
        });
    }
    
    // 按时间排序，最新的在前
    items.sort((a, b) => b.timestamp - a.timestamp);
    return items;
}

function extractTaskNameFromPath(loraPath) {
    if (!loraPath) return '';
    // 从路径中提取 train_xxx
    const parts = loraPath.split('/');
    for (const part of parts) {
        if (part.startsWith('train_')) {
            return part.slice(-12);
        }
    }
    return '';
}

function formatDateTime(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    return `${month}-${day} ${hours}:${minutes}`;
}

function renderInferenceHistory() {
    const container = document.getElementById('inference-history-list');
    if (!container) return;
    
    let items = allHistoryItems;
    
    // 搜索过滤
    if (historySearchKeyword) {
        const kw = historySearchKeyword.toLowerCase();
        items = items.filter(item => 
            item.displayName.toLowerCase().includes(kw) || 
            item.taskName.toLowerCase().includes(kw) ||
            (item.detail || '').toLowerCase().includes(kw)
        );
    }
    
    // 状态过滤
    if (historyFilterStatus !== 'all') {
        if (historyFilterStatus === 'running') {
            items = items.filter(item => ['pending', 'queued', 'loading', 'running'].includes(item.status));
        } else {
            items = items.filter(item => item.status === historyFilterStatus);
        }
    }
    
    if (items.length === 0) {
        const msg = historySearchKeyword || historyFilterStatus !== 'all' 
            ? '没有匹配的记录' 
            : '暂无历史记录';
        container.innerHTML = `<div class="empty-state small"><p>${msg}</p></div>`;
        updateHistoryLoadMore(0, 0);
        return;
    }
    
    // 分页
    const displayItems = items.slice(0, historyDisplayCount);
    
    let html = '';
    for (const item of displayItems) {
        // 删除按钮（所有非运行状态的条目都可删除）
        const deleteBtn = !item.isRunning ?
            `<button class="btn-delete-small" onclick="event.stopPropagation(); deleteInferenceTask('${item.taskId}', ${item.type === 'batch' ? `'${item.batchId}'` : 'null'})" title="删除任务">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
            </button>` : '';
        
        if (item.type === 'batch') {
            const batchRetryBtn = item.status === 'failed' ?
                `<button class="btn-retry-small" onclick="event.stopPropagation(); retryBatchInference('${item.batchId}')" title="重试失败任务">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                    </svg>
                </button>` : '';
            
            html += `
            <div class="history-item batch ${item.status}" onclick="viewInferenceResult('${item.taskId}', '${item.batchId}')">
                <div class="history-info">
                    <span class="history-name">${escapeHtml(item.taskName)}</span>
                    <span class="history-detail">${item.detail}</span>
                    <span class="history-time">${formatDateTime(item.createdAt)}</span>
                </div>
                <div class="history-actions">
                    <span class="history-status ${item.status}">${item.statusText}</span>
                    ${batchRetryBtn}
                    ${deleteBtn}
                </div>
            </div>`;
        } else {
            const cancelBtn = item.isRunning ? 
                `<button class="btn-cancel-small" onclick="event.stopPropagation(); cancelInferenceTask('${item.taskId}')" title="取消任务">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>` : '';
            
            const retryBtn = item.status === 'failed' ?
                `<button class="btn-retry-small" onclick="event.stopPropagation(); retryInferenceTask('${item.taskId}')" title="重新推理">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                    </svg>
                </button>` : '';
            
            html += `
            <div class="history-item ${item.status}" onclick="viewInferenceResult('${item.taskId}')">
                <div class="history-info">
                    <span class="history-name">${escapeHtml(item.taskName)}</span>
                    <span class="history-detail">${item.detail}</span>
                    <span class="history-time">${formatDateTime(item.createdAt)}</span>
                </div>
                <div class="history-actions">
                    <span class="history-status ${item.status}">${item.statusText}</span>
                    ${retryBtn}
                    ${cancelBtn}
                    ${deleteBtn}
                </div>
            </div>`;
        }
    }
    
    container.innerHTML = html;
    updateHistoryLoadMore(displayItems.length, items.length);
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function updateHistoryLoadMore(displayed, total) {
    const btn = document.getElementById('history-load-more');
    if (btn) {
        btn.style.display = displayed < total ? 'block' : 'none';
    }
}

function loadMoreInferenceHistory() {
    historyDisplayCount += 20;
    renderInferenceHistory();
}

function filterInferenceHistory() {
    historySearchKeyword = document.getElementById('history-search-input')?.value?.trim() || '';
    historyDisplayCount = 20;
    renderInferenceHistory();
}

function filterInferenceHistoryByStatus(status) {
    historyFilterStatus = status;
    historyDisplayCount = 20;
    // 更新 tab 激活状态
    document.querySelectorAll('.history-filter-tab').forEach(el => {
        el.classList.toggle('active', el.dataset.filter === status);
    });
    renderInferenceHistory();
}

function getInferenceStatusText(status) {
    const map = {
        pending: '等待LoRA',
        queued: '排队中',
        loading: '加载中',
        running: '推理中',
        completed: '已完成',
        failed: '失败'
    };
    return map[status] || status;
}

async function viewInferenceResult(taskId, batchId = null) {
    try {
        // 如果有 batchId，获取整个批次的任务
        if (batchId) {
            const res = await API.request(`/inference/tasks?batch_id=${batchId}`);
            if (res.code === 200 && res.data?.tasks) {
                const tasks = res.data.tasks;
                if (tasks.length > 1) {
                    // 重建批次状态用于网格展示
                    const epochs = [...new Set(tasks.map(t => t.lora_name))];
                    const imageCount = Math.max(...tasks.map(t => t.image_index || 0)) + 1;
                    
                    InferState.currentBatch = {
                        batchId,
                        epochs,
                        imageCount,
                        tasks,
                        visibleEpochs: new Set(epochs.map((_, i) => i))
                    };
                    
                    updateBatchResultArea();
                    return;
                }
            }
        }
        
        // 单个任务展示
        const res = await API.request(`/inference/task/${taskId}`);
        if (res.code === 200 && res.data) {
            InferState.currentBatch = null; // 清除批次状态
            updateResultArea(res.data.status, res.data);
        }
    } catch (e) {
        toast(`获取结果失败: ${e.message}`, 'error');
    }
}

async function cancelInferenceTask(taskId) {
    if (!confirm('确定要取消此推理任务吗？')) return;
    
    try {
        const res = await API.request(`/inference/task/${taskId}/stop`, { method: 'POST' });
        if (res.code === 200) {
            toast('任务已取消', 'success');
            loadInferenceHistory();
        } else {
            toast(res.message || '取消失败', 'error');
        }
    } catch (e) {
        toast(`取消失败: ${e.message}`, 'error');
    }
}

async function deleteInferenceTask(taskId, batchId) {
    // batchId 不为 null 时表示这是一个批次条目，需要删除整个批次
    if (batchId) {
        try {
            const res = await API.request(`/inference/tasks?batch_id=${batchId}`);
            if (res.code !== 200 || !res.data?.tasks) {
                toast('获取批次任务失败', 'error');
                return;
            }
            const taskCount = res.data.tasks.length;
            if (!confirm(`确定要删除此批次中的 ${taskCount} 个任务吗？\n删除后无法恢复。`)) return;
            
            const taskIds = res.data.tasks.map(t => t.task_id);
            const delRes = await API.request('/inference/tasks/bulk-delete', {
                method: 'POST',
                body: JSON.stringify({ task_ids: taskIds })
            });
            if (delRes.code === 200) {
                toast(`已删除 ${delRes.data?.deleted || taskCount} 个任务`, 'success');
                loadInferenceHistory();
            } else {
                toast(delRes.message || '删除失败', 'error');
            }
        } catch (e) {
            toast(`删除失败: ${e.message}`, 'error');
        }
        return;
    }
    
    // 单个任务删除
    if (!confirm('确定要删除此推理任务吗？\n删除后无法恢复。')) return;
    
    try {
        const res = await API.request(`/inference/task/${taskId}`, { method: 'DELETE' });
        if (res.code === 200) {
            toast('任务已删除', 'success');
            loadInferenceHistory();
        } else {
            toast(res.message || '删除失败', 'error');
        }
    } catch (e) {
        toast(`删除失败: ${e.message}`, 'error');
    }
}

async function retryInferenceTask(taskId) {
    if (!confirm('确定要重新推理此任务吗？')) return;
    
    try {
        const res = await API.request(`/inference/task/${taskId}/retry`, { method: 'POST' });
        if (res.code === 200) {
            toast('任务已重新排队', 'success');
            loadInferenceHistory();
        } else {
            toast(res.message || '重试失败', 'error');
        }
    } catch (e) {
        toast(`重试失败: ${e.message}`, 'error');
    }
}

async function retryBatchInference(batchId) {
    try {
        // 获取该批次所有任务
        const res = await API.request(`/inference/tasks?batch_id=${batchId}`);
        if (res.code !== 200 || !res.data?.tasks) {
            toast('获取批次任务失败', 'error');
            return;
        }
        
        const failedTasks = res.data.tasks.filter(t => t.status === 'failed');
        if (failedTasks.length === 0) {
            toast('该批次没有失败的任务', 'info');
            return;
        }
        
        const msg = `确定要重试该批次中 ${failedTasks.length} 个失败任务吗？\n任务将自动排队依次执行。`;
        if (!confirm(msg)) return;
        
        let successCount = 0;
        for (const task of failedTasks) {
            try {
                const retryRes = await API.request(`/inference/task/${task.task_id}/retry`, { method: 'POST' });
                if (retryRes.code === 200) successCount++;
            } catch (e) {
                console.error(`重试 ${task.task_id} 失败:`, e);
            }
        }
        
        if (successCount > 0) {
            toast(`${successCount} 个任务已重新排队`, 'success');
            loadInferenceHistory();
        } else {
            toast('所有重试均失败', 'error');
        }
    } catch (e) {
        toast(`重试失败: ${e.message}`, 'error');
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
            
            // 瀑布流布局：使用缩略图预加载，点击查看原图
            grid.innerHTML = res.data.images.map(img => {
                // 计算宽高比（默认为 1:1，如果没有尺寸信息）
                const ratio = img.width && img.height ? img.height / img.width : 1;
                return `
                <div class="gallery-card waterfall-item" 
                     style="--aspect-ratio: ${ratio};"
                     data-url="${img.url}"
                     data-name="${img.name}"
                     onclick="openImageLightbox('${img.url}', '${img.name}')">
                    <img src="${img.thumb_url}" 
                         alt="${img.name}" 
                         loading="lazy"
                         onerror="this.src='${img.url}'">
                    <div class="gallery-card-overlay">
                        <span class="name">${img.name}</span>
                        <button class="delete-btn" onclick="event.stopPropagation(); deleteGalleryImage('${img.id}')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            </svg>
                        </button>
                    </div>
                </div>
            `}).join('');
        }
    } catch (e) {
        grid.innerHTML = `<div class="empty-state error"><p>加载失败: ${e.message}</p></div>`;
    }
}

// Lightbox 原图查看
function openImageLightbox(url, name) {
    // 移除已存在的 lightbox
    const existing = document.getElementById('image-lightbox');
    if (existing) existing.remove();
    
    const lightbox = document.createElement('div');
    lightbox.id = 'image-lightbox';
    lightbox.className = 'lightbox-overlay';
    lightbox.innerHTML = `
        <div class="lightbox-content">
            <div class="lightbox-header">
                <span class="lightbox-title">${name}</span>
                <button class="lightbox-close" onclick="closeLightbox()">×</button>
            </div>
            <div class="lightbox-body">
                <div class="lightbox-loading">
                    <div class="spinner"></div>
                    <p>加载原图中...</p>
                </div>
                <img src="${url}" alt="${name}" onload="this.parentElement.querySelector('.lightbox-loading').style.display='none'; this.style.opacity=1;">
            </div>
        </div>
    `;
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) closeLightbox();
    });
    document.body.appendChild(lightbox);
    
    // ESC 键关闭
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeLightbox();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

function closeLightbox() {
    const lightbox = document.getElementById('image-lightbox');
    if (lightbox) lightbox.remove();
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
    
    // Drag & drop
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
        uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); handleAddVideos(e.dataTransfer.files); });
    }
    
    // 监听推理参数表单变化，自动保存状态
    const inferFormInputs = [
        'infer-prompt', 'infer-lora-strength', 'infer-width', 'infer-height',
        'infer-frames', 'infer-steps', 'infer-guidance', 'infer-seed', 'infer-auto-caption'
    ];
    inferFormInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', autoSaveState);
            if (el.type === 'text' || el.tagName === 'TEXTAREA') {
                el.addEventListener('input', autoSaveState);
            }
        }
    });
    
    // 监听 config tabs 变化
    document.querySelectorAll('.config-panel .tab').forEach(tab => {
        tab.addEventListener('click', autoSaveState);
    });
    
    // 触发词输入自动保存
    const twInput = document.getElementById('trigger-word');
    if (twInput) {
        twInput.addEventListener('input', autoSaveState);
        twInput.addEventListener('change', autoSaveState);
    }
    
    // 训练配置参数变化自动保存
    const trainingFormInputs = [
        'task-name', 'epochs-val', 'batch-size-val', 'grad-accum-val', 'grad-clip',
        'warmup-val', 'save-epochs-val', 'ckpt-epochs-val', 'clip-mode',
        'blocks-swap-val', 'act-ckpt-mode', 'model-dtype', 'transformer-dtype',
        'lora-rank-val', 'adapter-dtype', 'resolution', 'ar-bucket',
        'repeats-val', 'optimizer', 'lr', 'betas', 'weight-decay'
    ];
    trainingFormInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', autoSaveState);
            if (el.type === 'text' || el.type === 'number' || el.tagName === 'TEXTAREA') {
                el.addEventListener('input', autoSaveState);
            }
        }
    });
    // slider 滑块也要同步
    document.querySelectorAll('.slider-row .slider').forEach(slider => {
        slider.addEventListener('input', autoSaveState);
    });
    
    // Restore
    restorePage();
    
    console.log('Init complete');
});

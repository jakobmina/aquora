document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const btnEpoch = document.getElementById('btn-epoch');
    const btnReset = document.getElementById('btn-reset');
    const btnAuto = document.getElementById('btn-auto');
    const btnRoll = document.getElementById('btn-roll');
    
    const valEpoch = document.getElementById('val-epoch');
    const barEpoch = document.getElementById('bar-epoch');
    
    const valDrift = document.getElementById('val-drift');
    const barDrift = document.getElementById('bar-drift');
    
    const valGolden = document.getElementById('val-golden');
    
    const globalEpochs = document.getElementById('global-epochs');
    const globalEntropy = document.getElementById('global-entropy');
    const globalCoherence = document.getElementById('global-coherence');
    
    const barEmpuje = document.getElementById('bar-empuje');
    const barDrift072 = document.getElementById('bar-drift-072');
    const valDrift072 = document.getElementById('val-drift-072');
    
    const logContainer = document.getElementById('log-container');
    
    // Extracted Hex Elements
    const bigEndianSpan = document.querySelector('.big-endian .value');
    const littleEndianSpan = document.querySelector('.little-endian .value');
    
    let autoInterval = null;
    
    // API Calls
    async function fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            updateUI(data);
        } catch (e) {
            console.error("API connection failed", e);
        }
    }
    
    async function doEpoch() {
        try {
            const response = await fetch('/api/epoch', { method: 'POST' });
            const data = await response.json();
            updateUI(data);
        } catch (e) {
            console.error("Epoch failed", e);
        }
    }
    
    async function doReset() {
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const data = await response.json();
            updateUI(data);
        } catch (e) {
            console.error("Reset failed", e);
        }
    }
    
    async function doRoll() {
        try {
            const response = await fetch('/api/roll', { method: 'POST' });
            const data = await response.json();
            updateUI(data);
        } catch (e) {
            console.error("Roll failed", e);
        }
    }
    
    // Update UI
    function updateUI(data) {
        // O_N Master
        valGolden.innerText = data.golden_ratio.toFixed(4);
        
        // Progress
        valEpoch.innerText = `${data.epochs} / —`;
        barEpoch.style.width = `${Math.min(100, data.epochs * 2)}%`;
        
        valDrift.innerText = data.drift.toFixed(4);
        barDrift.style.width = `${Math.min(100, Math.abs(data.drift) * 1000)}%`;
        
        // Endianness values
        bigEndianSpan.innerText = data.hex_be;
        littleEndianSpan.innerText = data.hex_le;
        
        // Mahalanobis metrics (mapped for UI)
        let pctMah = Math.min(100, data.mahalanobis * 50);
        barEmpuje.style.width = `${pctMah}%`;
        
        // Using L_symp and L_metr for the visual drift comparison
        let l_ratio = data.L_metr > 0 ? (data.L_symp / data.L_metr) : 0;
        barDrift072.style.width = `${Math.min(100, l_ratio * 30)}%`;
        valDrift072.innerText = l_ratio.toFixed(4);
        
        // Global Metrics
        globalEpochs.innerText = data.epochs;
        globalEntropy.innerText = data.entropy.toFixed(3);
        
        let coherence = Math.max(0, 100 - data.entropy * 10).toFixed(0);
        globalCoherence.innerText = `${coherence}%`;
        
        // Logs
        logContainer.innerHTML = '';
        data.logs.forEach(log => {
            const el = document.createElement('div');
            el.className = 'log-entry';
            el.innerText = log;
            logContainer.appendChild(el);
        });
        logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    // Event Listeners
    btnEpoch.addEventListener('click', doEpoch);
    btnReset.addEventListener('click', doReset);
    btnRoll.addEventListener('click', doRoll);
    
    btnAuto.addEventListener('click', () => {
        if (autoInterval) {
            clearInterval(autoInterval);
            autoInterval = null;
            btnAuto.innerHTML = '<span class="icon text-yellow">⚡</span> auto x10';
        } else {
            autoInterval = setInterval(doEpoch, 500);
            btnAuto.innerHTML = '<span class="icon text-yellow">⏸</span> stop auto';
        }
    });
    
    // Initial load
    fetchMetrics();
});

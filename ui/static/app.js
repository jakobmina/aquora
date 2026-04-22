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
        // O_N Master & Particle Type
        valGolden.innerText = data.golden_ratio.toFixed(4);
        
        const typeBadge = document.getElementById('val-particle-type');
        typeBadge.innerText = data.particle_type.toUpperCase();
        if (data.particle_type === 'fermionic') {
            typeBadge.style.borderColor = 'var(--orange)';
            typeBadge.style.color = 'var(--orange)';
        } else {
            typeBadge.style.borderColor = 'var(--cyan)';
            typeBadge.style.color = 'var(--cyan)';
        }

        // Chirality
        const valChirality = document.getElementById('val-chirality');
        const barChirality = document.getElementById('bar-chirality');
        valChirality.innerText = data.chirality.toFixed(4);
        barChirality.style.width = `${Math.min(100, data.chirality * 100)}%`;
        
        // Vacuum Overlaps (W0-W3)
        if (data.pair_overlaps) {
            data.pair_overlaps.forEach((w, i) => {
                const valW = document.getElementById(`val-w${i}`);
                const barW = document.getElementById(`bar-w${i}`);
                if (valW) valW.innerText = w.toFixed(2);
                if (barW) {
                    // Map -2..2 to 0..100%
                    let pct = ((w + 2) / 4) * 100;
                    barW.style.width = `${Math.min(100, Math.max(0, pct))}%`;
                }
            });
        }

        // Metriplectic Analytics
        document.getElementById('val-lsymp').innerText = data.L_symp.toFixed(3);
        document.getElementById('val-lmetr').innerText = data.L_metr.toFixed(3);
        let ratio = data.L_metr !== 0 ? (data.L_symp / data.L_metr) : 0;
        document.getElementById('val-lratio').innerText = ratio.toFixed(3);

        // Drift
        valDrift.innerText = data.drift.toFixed(4);
        barDrift.style.width = `${Math.min(100, Math.abs(data.drift) * 100)}%`;
        
        // Endianness values
        bigEndianSpan.innerText = data.hex_be;
        littleEndianSpan.innerText = data.hex_le;
        
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

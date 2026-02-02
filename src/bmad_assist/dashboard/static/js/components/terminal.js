/**
 * Terminal component for output display
 * Uses xterm.js for full terminal emulation with ANSI color support
 */

window.terminalComponent = function() {
    return {
        // State
        output: [],
        autoScroll: true,
        terminalStatus: 'idle',

        _scrollTimeout: null,
        _validatorResetTimeout: null,

        // xterm.js instances
        _xterm: null,
        _fitAddon: null,
        _resizeObserver: null,

        // Output batching for performance
        _xtermWriteQueue: [],
        _xtermFlushTimer: null,
        _XTERM_FLUSH_INTERVAL: 50,

        // Validator progress tracking
        validatorProgress: {
            total: 0,
            completed: 0,
            failed: 0,
            validators: {}
        },

        // Terminal font size
        terminalFontSize: 13,
        terminalFontSizeMin: 9,
        terminalFontSizeMax: 24,

        // ANSI color codes for providers
        _providerColors: {
            opus: '\x1b[38;5;208m',
            gemini: '\x1b[38;5;39m',
            glm: '\x1b[38;5;40m',
            claude: '\x1b[38;5;141m',
            dashboard: '\x1b[38;5;245m',
            workflow: '\x1b[38;5;250m',
            bmad: '\x1b[38;5;250m'
        },
        _ANSI_RESET: '\x1b[0m',
        _ANSI_DIM: '\x1b[2m',

        initTerminal() {
            const savedFontSize = localStorage.getItem('bmad-terminal-font-size');
            if (savedFontSize) {
                const size = parseInt(savedFontSize, 10);
                if (size >= this.terminalFontSizeMin && size <= this.terminalFontSizeMax) {
                    this.terminalFontSize = size;
                }
            }
            this.$nextTick(() => this._initXterm());
        },

        _initXterm() {
            const container = this.$refs.xtermContainer;
            if (!container || this._xterm) return;

            this._xterm = new Terminal({
                theme: {
                    background: '#0a0a0f',
                    foreground: '#e4e4e7',
                    cursor: '#a855f7',
                    cursorAccent: '#0a0a0f',
                    selection: 'rgba(168, 85, 247, 0.3)',
                    black: '#18181b',
                    red: '#f87171',
                    green: '#4ade80',
                    yellow: '#facc15',
                    blue: '#60a5fa',
                    magenta: '#c084fc',
                    cyan: '#22d3ee',
                    white: '#e4e4e7',
                    brightBlack: '#52525b',
                    brightRed: '#fca5a5',
                    brightGreen: '#86efac',
                    brightYellow: '#fde047',
                    brightBlue: '#93c5fd',
                    brightMagenta: '#d8b4fe',
                    brightCyan: '#67e8f9',
                    brightWhite: '#fafafa'
                },
                fontSize: this.terminalFontSize,
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
                scrollback: 10000,
                cursorBlink: false,
                cursorStyle: 'bar',
                disableStdin: true,
                convertEol: true,
                allowProposedApi: true
            });

            this._fitAddon = new FitAddon.FitAddon();
            this._xterm.loadAddon(this._fitAddon);
            this._xterm.loadAddon(new WebLinksAddon.WebLinksAddon());
            this._xterm.open(container);
            this._fitAddon.fit();

            this._resizeObserver = new ResizeObserver(() => {
                if (this._fitAddon) this._fitAddon.fit();
            });
            this._resizeObserver.observe(container);

            this._xterm.writeln('\x1b[38;5;141m' + '='.repeat(60) + this._ANSI_RESET);
            this._xterm.writeln('\x1b[38;5;141m  bmad-assist dashboard' + this._ANSI_RESET);
            this._xterm.writeln('\x1b[38;5;245m  Terminal ready. Click Start to begin loop.' + this._ANSI_RESET);
            this._xterm.writeln('\x1b[38;5;141m' + '='.repeat(60) + this._ANSI_RESET);
            this._xterm.writeln('');
        },

        addOutput(data) {
            const time = new Date(data.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false });
            const line = {
                time,
                provider: data.provider,
                text: data.line
            };
            this.output.push(line);

            if (this.output.length > 1000) {
                this.output = this.output.slice(-500);
            }

            if (data.provider === 'dashboard' && data.line.includes('Loop ended')) {
                this.loopRunning = false;
                this.pauseRequested = false;
            }

            this._queueXtermWrite(line);
        },

        _queueXtermWrite(line) {
            if (!this._xterm) return;

            // Log level filter (frontend filtering for instant response)
            // Dashboard messages always shown, workflow output filtered
            if (line.provider === 'workflow' && this.shouldShowLogLine && !this.shouldShowLogLine(line.text)) {
                return;
            }

            const formatted = this._formatXtermLine(line);
            this._xtermWriteQueue.push(formatted);

            if (!this._xtermFlushTimer) {
                this._xtermFlushTimer = setTimeout(() => this._flushXtermQueue(), this._XTERM_FLUSH_INTERVAL);
            }
        },

        _flushXtermQueue() {
            if (!this._xterm || this._xtermWriteQueue.length === 0) {
                this._xtermFlushTimer = null;
                return;
            }
            const batch = this._xtermWriteQueue.join('\r\n') + '\r\n';
            this._xterm.write(batch);
            this._xtermWriteQueue = [];
            this._xtermFlushTimer = null;
            if (this.autoScroll) this._xterm.scrollToBottom();
        },

        _formatXtermLine(line) {
            const provider = line.provider || 'bmad';
            const providerColor = this._providerColors[provider] || this._providerColors.bmad;
            return `${this._ANSI_DIM}[${line.time}]${this._ANSI_RESET} ${providerColor}[${provider}]${this._ANSI_RESET} ${line.text}`;
        },

        scrollToBottom() {
            if (this._xterm) {
                this._xterm.scrollToBottom();
                this.autoScroll = true;
            }
        },

        adjustTerminalFontSize(delta) {
            const newSize = this.terminalFontSize + delta;
            if (newSize >= this.terminalFontSizeMin && newSize <= this.terminalFontSizeMax) {
                this.terminalFontSize = newSize;
                localStorage.setItem('bmad-terminal-font-size', newSize);
                if (this._xterm) {
                    this._xterm.options.fontSize = newSize;
                    if (this._fitAddon) this._fitAddon.fit();
                }
            }
        },

        handleTerminalWheel(e) {
            if (e.ctrlKey) {
                e.preventDefault();
                this.adjustTerminalFontSize(e.deltaY < 0 ? 1 : -1);
            }
        },

        handleTerminalKeydown(e) {
            if (e.ctrlKey && (e.key === '+' || e.key === '=' || e.key === 'NumpadAdd')) {
                e.preventDefault();
                this.adjustTerminalFontSize(1);
            } else if (e.ctrlKey && (e.key === '-' || e.key === 'NumpadSubtract')) {
                e.preventDefault();
                this.adjustTerminalFontSize(-1);
            } else if (e.ctrlKey && e.key === '0') {
                e.preventDefault();
                this.terminalFontSize = 13;
                localStorage.setItem('bmad-terminal-font-size', 13);
                if (this._xterm) {
                    this._xterm.options.fontSize = 13;
                    if (this._fitAddon) this._fitAddon.fit();
                }
            }
        },

        clearTerminal() {
            if (this._xterm) this._xterm.clear();
            this.output = [];
        },

        _handleValidatorProgress(data) {
            const { validator_id, status, duration_ms } = data.data || {};
            if (!validator_id) return;
            if (this._validatorResetTimeout) {
                clearTimeout(this._validatorResetTimeout);
                this._validatorResetTimeout = null;
            }
            this.validatorProgress.validators[validator_id] = { status, duration_ms };
            if (status === 'completed') {
                this.validatorProgress.completed++;
            } else if (status === 'timeout' || status === 'failed') {
                this.validatorProgress.failed++;
            }
            this.validatorProgress.total = Object.keys(this.validatorProgress.validators).length;
            console.debug(`Validator ${validator_id}: ${status}`,
                `(${this.validatorProgress.completed}/${this.validatorProgress.total})`);
        },

        _handlePhaseComplete(data) {
            const { phase_name, success, validator_count, failed_count } = data.data || {};
            if (this._validatorResetTimeout) {
                clearTimeout(this._validatorResetTimeout);
                this._validatorResetTimeout = null;
            }
            this.validatorProgress.total = validator_count || this.validatorProgress.total;
            this.validatorProgress.failed = failed_count || this.validatorProgress.failed;
            this.validatorProgress.completed = validator_count - failed_count;
            console.log(`Phase ${phase_name} complete:`,
                `${this.validatorProgress.completed}/${this.validatorProgress.total} succeeded`,
                success ? '(SUCCESS)' : '(FAILED)');
            this._validatorResetTimeout = setTimeout(() => {
                this.validatorProgress = { total: 0, completed: 0, failed: 0, validators: {} };
                this._validatorResetTimeout = null;
            }, 3000);
        },

        get validatorProgressPercent() {
            if (this.validatorProgress.total === 0) return 0;
            const done = this.validatorProgress.completed + this.validatorProgress.failed;
            return Math.round((done / this.validatorProgress.total) * 100);
        },

        get isValidating() {
            return this.validatorProgress.total > 0 &&
                (this.validatorProgress.completed + this.validatorProgress.failed) < this.validatorProgress.total;
        },

        _destroyXterm() {
            if (this._xtermFlushTimer) {
                clearTimeout(this._xtermFlushTimer);
                this._xtermFlushTimer = null;
            }
            if (this._resizeObserver) {
                this._resizeObserver.disconnect();
                this._resizeObserver = null;
            }
            if (this._xterm) {
                this._xterm.dispose();
                this._xterm = null;
            }
            this._fitAddon = null;
        }
    };
};

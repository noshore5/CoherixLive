from flask import Flask, render_template, jsonify, Response
from utils.app_utils import get_binance_ohlcv, transform, coherence
from plot_coherence import plot_coherence
import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
#
@app.route('/api/ohlcv/<symbol>')
def get_ohlcv(symbol):
    # symbol: e.g. 'BTCUSDT' or 'ETHUSDT'
    data = get_binance_ohlcv(symbol)
    return jsonify(data)


@app.route('/api/wavelet_coherence')
def get_wavelet_coherence():
    # Get last 1 hour of BTCUSDT and ETHUSDT
    btc = get_binance_ohlcv('BTCUSDT')[-3600:]
    eth = get_binance_ohlcv('ETHUSDT')[-3600:]
    if len(btc) < 10 or len(eth) < 10:
        return 'Not enough data', 400
    btc_close = np.array([d['close'] for d in btc])
    eth_close = np.array([d['close'] for d in eth])
    data = plot_coherence(btc_close, eth_close, fs=1)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import requests
import yfinance as yf
import pandas as pd

# Configuração
SYMBOL = 'AAPL'  # Altere para TSLA, PETR4.SA, etc.
API_URL = "http://localhost:8000/predict"

print("=" * 60)
print(f"TESTANDO API COM {SYMBOL}")
print("=" * 60)

# 1. Baixar dados históricos
print(f"\n📥 Baixando últimos 90 dias de {SYMBOL}...")
df = yf.download(SYMBOL, period='3mo', progress=False, auto_adjust=False)

if df.empty:
    print(f"❌ Erro: Não foi possível baixar dados para {SYMBOL}")
    exit(1)

# 2. Preparar dados no formato correto [Open, High, Low, Close, Volume, Adj Close]
print(f"✓ Dados baixados: {len(df)} dias")

# Verificar se existe Adj Close (algumas versões do yfinance podem não retornar)
if 'Adj Close' not in df.columns:
    print("⚠️  Adj Close não encontrado, usando Close como Adj Close")
    df['Adj Close'] = df['Close']

# Pegar últimos 60 dias
last_60 = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].tail(60)

if len(last_60) < 60:
    print(f"❌ Erro: Apenas {len(last_60)} dias disponíveis, necessário 60")
    exit(1)

# 3. Converter para formato de lista de listas
payload = {
    "last_60_days": last_60.values.tolist()
}

print(f"✓ Preparados {len(payload['last_60_days'])} dias")
print(f"\n📊 Exemplo de dados (último dia):")
last_day = last_60.iloc[-1]
print(f"   Open:       ${last_day['Open'].item():.2f}")
print(f"   High:       ${last_day['High'].item():.2f}")
print(f"   Low:        ${last_day['Low'].item():.2f}")
print(f"   Close:      ${last_day['Close'].item():.2f}")
print(f"   Volume:     {last_day['Volume'].item():,.0f}")
print(f"   Adj Close:  ${last_day['Adj Close'].item():.2f}")

# 4. Fazer requisição
print(f"\n🚀 Enviando requisição para {API_URL}...")

try:
    response = requests.post(API_URL, json=payload, timeout=15)
    response.raise_for_status()
    
    result = response.json()
    
    print("\n✅ RESPOSTA DA API:")
    print("=" * 60)
    current_price = last_60.iloc[-1]['Close'].item()
    print(f"   Previsão:      ${result['prediction']:.2f}")
    print(f"   Preço atual:   ${current_price:.2f}")
    
    diff = result['prediction'] - current_price
    diff_pct = (diff / current_price) * 100
    
    print(f"   Variação:      ${diff:+.2f} ({diff_pct:+.2f}%)")
    print("=" * 60)
    
except requests.exceptions.ConnectionError:
    print("\n❌ Erro: API não está rodando!")
    print("   Execute em outro terminal: make run-local")
except requests.exceptions.HTTPError as e:
    print(f"\n❌ Erro HTTP {e.response.status_code}:")
    print(f"   {e.response.json()}")
except Exception as e:
    print(f"\n❌ Erro: {e}")
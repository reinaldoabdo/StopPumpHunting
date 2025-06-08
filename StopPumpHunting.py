# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
from typing import Dict, List, Optional, Tuple
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)

class CryptoHuntingStrategy(IStrategy):
    """
    Estratégia avançada para exploração de armadilhas de mercado e proteção contra black swans
    
    Características principais:
    - Detecção de falsos rompimentos (stop hunting)
    - Exploração de preços artificialmente inflados (pumps)
    - Proteção robusta contra colapsos de mercado
    - Gerenciamento de risco dinâmico
    """
    
    INTERFACE_VERSION = 3
    
    # Configurações básicas
    timeframe = '15m'
    stoploss = -0.10  # Backup stoploss
    use_custom_stoploss = True
    position_adjustment_enable = True
    can_short = True
    
    # Configuração de alavancagem máxima
    max_leverage = 3.0
    
    # Configurações de capital
    max_position_size = 0.02  # 2% do capital por trade
    max_total_exposure = 0.20  # 20% exposição total
    
    # Parâmetros otimizáveis
    pivot_period = IntParameter(15, 45, default=30, space="buy")
    rsi_overbought = IntParameter(75, 85, default=80, space="sell")
    rsi_oversold = IntParameter(15, 35, default=30, space="buy")
    volume_multiplier_long = DecimalParameter(1.5, 3.0, default=1.5, space="buy")
    volume_multiplier_short = DecimalParameter(2.0, 4.0, default=3.0, space="sell")
    atr_sl_multiplier_long = DecimalParameter(0.5, 1.5, default=0.75, space="buy")
    atr_sl_multiplier_short = DecimalParameter(0.5, 1.5, default=1.0, space="sell")
    black_swan_trigger = DecimalParameter(-15, -7, default=-10, space="protection")
    
    # Configurações de proteção
    circuit_breaker_volume_drop = 0.40  # 40% queda no volume
    circuit_breaker_spread = 0.005  # 0.5% spread máximo
    circuit_breaker_drawdown = 0.15  # 15% drawdown diário
    
    # Configurações de take profit
    tp1_ratio = 0.50  # 50% da posição no TP1
    tp2_ratio = 0.25  # 25% da posição no TP2
    tp3_ratio = 0.25  # 25% da posição no TP3
    
    # Proteções integradas
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period": 24,
                "trade_limit": 4,
                "stop_duration": 12,
                "max_allowed_drawdown": 0.15
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,
                "trade_limit": 2,
                "stop_duration_candles": 6,
                "required_profit": -0.05
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                 side: str, **kwargs) -> float:
        """
        Determina alavancagem baseada na estabilidade do par
        """
        # Pares mais estáveis podem usar alavancagem maior
        stable_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        if pair in stable_pairs:
            return min(self.max_leverage, max_leverage)
        else:
            return min(2.0, max_leverage)  # Máximo 2x para pares menos estáveis

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stop loss dinâmico baseado em ATR e mínimos/máximos locais
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) < 20:
            return self.stoploss
        
        current_candle = dataframe.iloc[-1]
        atr = current_candle['atr']
        
        if trade.is_short:
            # Para shorts, usar máximos dos últimos 5 períodos
            recent_high = dataframe['high'].tail(5).max()
            dynamic_sl = (recent_high - current_rate) / current_rate
            atr_adjustment = atr * self.atr_sl_multiplier_short.value / current_rate
            return -(dynamic_sl + atr_adjustment)
        else:
            # Para longs, usar mínimos dos últimos 5 períodos
            recent_low = dataframe['low'].tail(5).min()
            dynamic_sl = (current_rate - recent_low) / current_rate
            atr_adjustment = atr * self.atr_sl_multiplier_long.value / current_rate
            return -(dynamic_sl + atr_adjustment)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adiciona todos os indicadores necessários
        """
        # Pivot Points
        dataframe['pivot_point'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['s1'] = (2 * dataframe['pivot_point']) - dataframe['high']
        dataframe['r1'] = (2 * dataframe['pivot_point']) - dataframe['low']
        dataframe['s2'] = dataframe['pivot_point'] - (dataframe['high'] - dataframe['low'])
        dataframe['r2'] = dataframe['pivot_point'] + (dataframe['high'] - dataframe['low'])
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        
        # Volume médio
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # ATR para gestão de risco
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Detecção de divergências
        dataframe['price_change'] = dataframe['close'].pct_change(5)
        dataframe['rsi_change'] = dataframe['rsi'].pct_change(5)
        dataframe['bearish_divergence'] = (
            (dataframe['price_change'] > 0) & 
            (dataframe['rsi_change'] < 0) & 
            (dataframe['rsi'] > 70)
        )
        
        # Detecção de expansão abrupta das Bollinger Bands
        dataframe['bb_expansion'] = dataframe['bb_width'].pct_change(3) > 0.20
        
        # Monitoramento de black swan events
        dataframe['price_change_5m'] = dataframe['close'].pct_change(1)  # Aproximação para 5min em timeframe 15m
        dataframe['volume_spike'] = dataframe['volume_ratio'] > 5.0
        dataframe['black_swan_signal'] = (
            (dataframe['price_change_5m'] < self.black_swan_trigger.value / 100) |
            (dataframe['volume_spike'] & (dataframe['price_change_5m'] < -0.05))
        )
        
        # Níveis de suporte e resistência dinâmicos
        dataframe['support_level'] = dataframe['low'].rolling(window=20).min()
        dataframe['resistance_level'] = dataframe['high'].rolling(window=20).max()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sinaliza entradas baseadas em falsos rompimentos e preços inflados
        """
        # Condições para LONG (Falsos rompimentos de suporte)
        long_conditions = (
            # Preço violou S1 mas fechou acima
            (dataframe['low'] < dataframe['s1']) &
            (dataframe['close'] > dataframe['s1']) &
            
            # Volume acima da média
            (dataframe['volume_ratio'] > self.volume_multiplier_long.value) &
            
            # RSI oversold
            (dataframe['rsi'] < self.rsi_oversold.value) &
            
            # MACD histograma positivo (momentum)
            (dataframe['macd_hist'] > 0) &
            
            # Não há black swan event
            (~dataframe['black_swan_signal'])
        )
        
        # Condições para SHORT (Preços artificialmente inflados)
        short_conditions = (
            # RSI overbought
            (dataframe['rsi'] > self.rsi_overbought.value) &
            
            # Preço acima da banda superior de Bollinger
            (dataframe['close'] > dataframe['bb_upper']) &
            
            # Volume muito alto (pump artificial)
            (dataframe['volume_ratio'] > self.volume_multiplier_short.value) &
            
            # MACD negativo (momentum bearish)
            (dataframe['macd'] < 0) &
            
            # Divergência bearish ou expansão abrupta das bandas
            (dataframe['bearish_divergence'] | dataframe['bb_expansion']) &
            
            # Não há black swan event
            (~dataframe['black_swan_signal'])
        )
        
        dataframe.loc[long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sinaliza saídas baseadas em take profits escalonados
        """
        # Saída LONG
        long_exit_conditions = (
            # TP1: Preço atingiu R1
            (dataframe['close'] >= dataframe['r1']) |
            
            # Reversão do momentum
            ((dataframe['rsi'] > 70) & (dataframe['macd_hist'] < 0)) |
            
            # Black swan event
            (dataframe['black_swan_signal'])
        )
        
        # Saída SHORT
        short_exit_conditions = (
            # TP1: Preço atingiu S1
            (dataframe['close'] <= dataframe['s1']) |
            
            # RSI voltou ao normal
            (dataframe['rsi'] < 50) |
            
            # Black swan event
            (dataframe['black_swan_signal'])
        )
        
        dataframe.loc[long_exit_conditions, 'exit_long'] = 1
        dataframe.loc[short_exit_conditions, 'exit_short'] = 1
        
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Lógica de saída customizada para take profits escalonados
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) < 2:
            return None
        
        current_candle = dataframe.iloc[-1]
        
        # Verificar black swan events
        if current_candle['black_swan_signal']:
            return "black_swan_protection"
        
        # Take profit escalonado para longs
        if not trade.is_short:
            if current_rate >= current_candle['r1'] and current_profit > 0.02:
                return "tp1_r1"
            elif current_rate >= current_candle['r2'] and current_profit > 0.04:
                return "tp2_r2"
        
        # Take profit escalonado para shorts
        else:
            if current_rate <= current_candle['s1'] and current_profit > 0.02:
                return "tp1_s1"
            elif current_rate <= current_candle['s2'] and current_profit > 0.04:
                return "tp2_s2"
        
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Confirma entrada do trade com verificações adicionais de segurança
        """
        # Verificar se o par atende aos critérios mínimos de liquidez
        if not self._check_pair_liquidity(pair):
            logger.warning(f"Par {pair} não atende critérios de liquidez mínima")
            return False
        
        # Verificar exposição total
        if self._get_total_exposure() + (amount * rate) > self.max_total_exposure:
            logger.warning(f"Exposição total seria excedida com este trade")
            return False
        
        # Verificar se não há eventos macro próximos (placeholder)
        if self._is_macro_event_near(current_time):
            logger.warning(f"Evento macro próximo, evitando novo trade")
            return False
        
        return True

    def _check_pair_liquidity(self, pair: str) -> bool:
        """
        Verifica se o par tem liquidez suficiente
        """
        try:
            ticker = self.dp.ticker(pair)
            if ticker:
                # Verificar volume 24h > $10 milhões (aproximação)
                volume_24h = ticker.get('quoteVolume', 0)
                return volume_24h > 10_000_000
        except Exception as e:
            logger.error(f"Erro ao verificar liquidez do par {pair}: {e}")
        
        return False

    def _get_total_exposure(self) -> float:
        """
        Calcula exposição total atual
        """
        try:
            total_exposure = 0.0
            for trade in Trade.get_open_trades():
                total_exposure += trade.amount * trade.open_rate
            return total_exposure
        except Exception:
            return 0.0

    def _is_macro_event_near(self, current_time: datetime) -> bool:
        """
        Verifica se há eventos macro próximos (placeholder)
        """
        # Implementar lógica para detectar eventos como Fed meetings, CPI, etc.
        # Por enquanto, retorna False
        return False

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Determina quantidade de stake baseada na gestão de risco
        """
        # Calcular 2% do capital total
        wallet_balance = self.wallets.get_total_stake_amount()
        max_stake_amount = wallet_balance * self.max_position_size
        
        # Usar o menor valor entre proposto e máximo permitido
        return min(proposed_stake, max_stake_amount, max_stake)

    def version(self) -> str:
        """
        Retorna versão da estratégia
        """
        return "CryptoHuntingStrategy v2.1 - Stop Hunting & Pump Exploitation"

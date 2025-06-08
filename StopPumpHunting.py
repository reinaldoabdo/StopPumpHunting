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
    opportunity_mode = CategoricalParameter(
        ["conservative", "balanced", "aggressive"], 
        default="balanced", space="buy"
    )
    
    # Configurações de proteção
    circuit_breaker_volume_drop = 0.40  # 40% queda no volume
    circuit_breaker_spread = 0.005  # 0.5% spread máximo
    circuit_breaker_drawdown = 0.15  # 15% drawdown diário
    
    # Configurações de take profit
    tp1_ratio = 0.50  # 50% da posição no TP1
    tp2_ratio = 0.25  # 25% da posição no TP2
    tp3_ratio = 0.25  # 25% da posição no TP3
    
    # Trailing stop (ativado apenas após TP1)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Parâmetros adicionais
    trailing_activation_pct = DecimalParameter(0.01, 0.05, default=0.015, space="sell")
    liquidity_threshold = DecimalParameter(0.3, 0.7, default=0.5, space="protection")
    
    # Eventos macro para evitar
    macro_events = [
        datetime(2024, 12, 18, 14, 30),  # FOMC Meeting
        datetime(2025, 1, 15, 13, 30),   # CPI Release
        datetime(2025, 2, 12, 14, 30),   # FOMC Meeting
        datetime(2025, 3, 12, 13, 30),   # CPI Release
    ]
    
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
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) & # Preço subindo
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['rsi'].shift(1) < dataframe['rsi'].shift(2)) & # RSI caindo
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
            (dataframe['rsi'] > 70)
        )
        # Correção da lógica de bearish_divergence (original parecia ter um erro)
        # A divergência bearish clássica é: Preço faz um topo mais alto, RSI faz um topo mais baixo.
        # Simplificando para detecção em N períodos:
        dataframe['bearish_divergence_corrected'] = (
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

        # Indicador de Sensibilidade Dinâmica
        dataframe['volatility_index'] = dataframe['atr'] / dataframe['close']
        dataframe['activity_ratio'] = dataframe['volume'] / dataframe['volume'].rolling(window=100, min_periods=20).mean().fillna(1) # Evitar NaN/ZeroDivision

        # Auto-ajuste de parâmetros (RSI dinâmico)
        # Criamos uma coluna para o threshold dinâmico do RSI
        dataframe['dynamic_rsi_oversold'] = self.rsi_oversold.value # Valor padrão
        dataframe.loc[dataframe['activity_ratio'] > 1.2, 'dynamic_rsi_oversold'] = 35  # Mais sensível
        dataframe.loc[dataframe['activity_ratio'] <= 1.2, 'dynamic_rsi_oversold'] = self.rsi_oversold.value # Mais rigoroso (ou valor base)

        # Market Pulse Monitor
        dataframe['market_pulse'] = dataframe.apply(lambda row: self._market_pulse(row), axis=1)
        
        # Heatmap de liquidações (opcional)
        try:
            # Placeholder para dados de liquidações - implementar com API externa
            dataframe['liquidations'] = 0
            # liquidation_data = self.dp.get_liquidations(pair, self.timeframe)
            # dataframe = self._merge_liquidation_data(dataframe, liquidation_data)
        except Exception as e:
            logger.warning(f"Não foi possível obter dados de liquidações para {metadata['pair']}: {e}")
            dataframe['liquidations'] = 0
        
        # Inicializar colunas de entrada para evitar erros em métricas de performance
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        # Métricas de performance tracking
        dataframe.loc[:, 'false_breakout_success'] = (
            (dataframe['enter_long'] == 1) & 
            (dataframe['close'].shift(-3) > dataframe['r1'])
        ).fillna(False)
        
        dataframe.loc[:, 'pump_exploit_success'] = (
            (dataframe['enter_short'] == 1) & 
            (dataframe['close'].shift(-3) < dataframe['s1'])
        ).fillna(False)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sinaliza entradas baseadas em falsos rompimentos e preços inflados
        """
        # Inicializar signal_score
        dataframe['signal_score'] = 0

        # Condições para LONG (Falsos rompimentos de suporte)
        base_long_conditions = (
            # Preço violou S1 mas fechou acima
            (dataframe['low'] < dataframe['s1']) &
            (dataframe['close'] > dataframe['s1']) &
            
            # Volume acima da média
            (dataframe['volume_ratio'] > self.volume_multiplier_long.value) &

            # RSI oversold (usando o threshold dinâmico)
            (dataframe['rsi'] < dataframe['dynamic_rsi_oversold']) &

            # MACD histograma positivo (momentum)
            (dataframe['macd_hist'] > 0) &
            
            # Não há black swan event
            (~dataframe['black_swan_signal'])
        )
        dataframe.loc[base_long_conditions, 'signal_score'] += 30 # rsi_oversold
        dataframe.loc[base_long_conditions & (dataframe['volume_ratio'] > self.volume_multiplier_long.value), 'signal_score'] += 25 # volume_spike
        dataframe.loc[base_long_conditions & (dataframe['close'] > dataframe['s1']), 'signal_score'] += 45 # pivot_confirmation (S1)

        # Condições adicionais para modo AGGRESSIVE
        aggressive_long_conditions = pd.Series(False, index=dataframe.index)
        if self.opportunity_mode.value == "aggressive":
            logger.info(f"Modo AGGRESSIVE ativo para {metadata['pair']}")
            aggressive_long_conditions = (
                (dataframe['volume'] > dataframe['volume'].shift(1) * 1.3) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['rsi'] < 45) # RSI um pouco menos restritivo para agressivo
            )
            dataframe.loc[aggressive_long_conditions, 'signal_score'] += 20 # Aggressive bonus

        # Condições para Range Trading (entrada LONG)
        range_long_conditions = (
            (dataframe['bb_width'] < 0.05) &  # Bandas estreitas indicam range
            (dataframe['rsi'].between(40, 60)) & # RSI em zona neutra
            (qtpylib.crossed_above(dataframe['close'], dataframe['bb_lower'])) & # Cruzou acima da banda inferior
            (~dataframe['black_swan_signal'])
        )
        dataframe.loc[range_long_conditions, 'signal_score'] += 35 # Range trading bonus

        # Combinar condições de LONG
        final_long_conditions = (
            base_long_conditions | (aggressive_long_conditions & base_long_conditions) | range_long_conditions
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
            (dataframe['bearish_divergence_corrected'] | dataframe['bb_expansion']) &
            
            # Liquidações altas (heatmap)
            (dataframe['liquidations'] > 0) &
            
            # Não há black swan event
            (~dataframe['black_swan_signal'])
        )
        
        dataframe.loc[final_long_conditions, 'enter_long'] = 1
        dataframe.loc[short_conditions, 'enter_short'] = 1

        # Log de modo operacional e pulso de mercado para trades potenciais
        if final_long_conditions.any() or short_conditions.any():
            logger.info(f"Pair: {metadata['pair']}, Mode: {self.opportunity_mode.value}, Market Pulse: {dataframe['market_pulse'].iloc[-1] if not dataframe.empty else 'N/A'}")
            logger.info(f"Pair: {metadata['pair']}, Scores (last): {dataframe['signal_score'].iloc[-1] if not dataframe.empty else 'N/A'}")
        
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
                   current_rate: float, current_profit: float, **kwargs) -> Optional[Tuple[str, float]]:
        """
        Lógica de saída customizada para take profits escalonados
        Retorna: (exit_reason, exit_percentage) ou None
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) < 2:
            return None
        
        current_candle = dataframe.iloc[-1]
        
        # Verificar black swan events - saída total imediata
        if current_candle['black_swan_signal']:
            logger.critical(f"BLACK SWAN EVENT detectado para {pair} - Saída imediata total")
            return ("black_swan_protection", 1.0)
        
        # Gerenciar take profits escalonados
        if not trade.is_short:
            # LONG positions
            if current_rate >= current_candle['r1'] and current_profit > 0.015:
                # TP1: Fechar 50% da posição
                if not hasattr(trade, 'tp1_executed') or not trade.tp1_executed:
                    logger.info(f"TP1 atingido para {pair} - Fechando {self.tp1_ratio*100}%")
                    trade.tp1_executed = True
                    # Ativar trailing stop para posição restante
                    self._activate_trailing_stop(trade)
                    return ("tp1_r1", self.tp1_ratio)
                    
            elif current_rate >= current_candle['r2'] and current_profit > 0.035:
                # TP2: Fechar mais 25% da posição
                if hasattr(trade, 'tp1_executed') and not hasattr(trade, 'tp2_executed'):
                    logger.info(f"TP2 atingido para {pair} - Fechando {self.tp2_ratio*100}%")
                    trade.tp2_executed = True
                    return ("tp2_r2", self.tp2_ratio)
                    
        else:
            # SHORT positions
            if current_rate <= current_candle['s1'] and current_profit > 0.015:
                # TP1: Fechar 50% da posição
                if not hasattr(trade, 'tp1_executed') or not trade.tp1_executed:
                    logger.info(f"TP1 SHORT atingido para {pair} - Fechando {self.tp1_ratio*100}%")
                    trade.tp1_executed = True
                    # Ativar trailing stop para posição restante
                    self._activate_trailing_stop(trade)
                    return ("tp1_s1", self.tp1_ratio)
                    
            elif current_rate <= current_candle['s2'] and current_profit > 0.035:
                # TP2: Fechar mais 25% da posição
                if hasattr(trade, 'tp1_executed') and not hasattr(trade, 'tp2_executed'):
                    logger.info(f"TP2 SHORT atingido para {pair} - Fechando {self.tp2_ratio*100}%")
                    trade.tp2_executed = True
                    return ("tp2_s2", self.tp2_ratio)
        
        return None

    def _activate_trailing_stop(self, trade: Trade) -> None:
        """
        Ativa trailing stop após TP1 ser executado
        """
        try:
            # Definir trailing stop para a posição restante (25% + 25% = 50%)
            trade.trailing_stop = True
            trade.trailing_stop_positive = self.trailing_activation_pct.value
            trade.trailing_only_offset_is_reached = True
            logger.info(f"Trailing stop ativado para {trade.pair} com offset {self.trailing_activation_pct.value}")
        except Exception as e:
            logger.error(f"Erro ao ativar trailing stop para {trade.pair}: {e}")

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
        
        # Verificar liquidez do order book
        if not self._check_orderbook_liquidity(pair, amount):
            logger.warning(f"Liquidez insuficiente no order book para {pair}")
            return False
        
        # Verificar exposição total
        if self._get_total_exposure() + (amount * rate) > self.max_total_exposure:
            logger.warning(f"Exposição total seria excedida com este trade")
            return False
        
        # Verificar se não há eventos macro próximos
        if self._is_macro_event_near(current_time):
            logger.warning(f"Evento macro próximo, evitando novo trade para {pair}")
            return False
        
        logger.info(f"Trade confirmado para {pair} - {side} - Valor: {amount * rate:.2f}")
        return True

    def _check_orderbook_liquidity(self, pair: str, amount: float) -> bool:
        """
        Verifica liquidez do order book antes de entrar no trade
        """
        try:
            order_book = self.dp.orderbook(pair, 1)
            if order_book and 'bids' in order_book and 'asks' in order_book:
                if order_book['bids'] and order_book['asks']:
                    bid_volume = order_book['bids'][0][1] if order_book['bids'] else 0
                    ask_volume = order_book['asks'][0][1] if order_book['asks'] else 0
                    
                    # Verificar se há pelo menos 50% do volume necessário disponível
                    required_volume = amount * self.liquidity_threshold.value
                    
                    if bid_volume < required_volume or ask_volume < required_volume:
                        logger.warning(f"Volume insuficiente no order book: bid={bid_volume}, ask={ask_volume}, necessário={required_volume}")
                        return False
                    
                    return True
        except Exception as e:
            logger.error(f"Erro ao verificar order book para {pair}: {e}")
        
        return False

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
        Verifica se há eventos macro próximos (6 horas antes/depois)
        """
        try:
            for event in self.macro_events:
                # Verificar se estamos dentro de 6 horas do evento
                time_diff = abs((current_time - event).total_seconds())
                if time_diff < 6 * 3600:  # 6 horas em segundos
                    logger.info(f"Evento macro próximo detectado: {event}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar eventos macro: {e}")
            return False

    def _market_pulse(self, row: pd.Series) -> str:
        """
        Classifica o mercado baseado em volatilidade e volume (usado em apply).
        """
        # Usar valores da linha (row) que já foram calculados
        volatility = row.get('volatility_index', 0)
        activity = row.get('activity_ratio', 0)

        # Definir thresholds (podem ser otimizados ou ajustados)
        # Estes são exemplos, ajuste conforme necessário
        if volatility > 0.005 and activity > 1.5: # Ex: ATR > 0.5% do preço e volume 50% acima da média
            return "HIGH_ENERGY"
        elif volatility < 0.0015 or activity < 0.7: # Ex: ATR < 0.15% do preço ou volume 30% abaixo da média
            return "LOW_ENERGY"
        else:
            return "NORMAL"

    def _merge_liquidation_data(self, dataframe: DataFrame, liquidation_data: dict) -> DataFrame:
        """
        Mescla dados de liquidações com o dataframe principal
        """
        try:
            # Implementação placeholder - substituir com API real de liquidações
            # Exemplo: Binance Futures API, Coinglass, etc.
            if liquidation_data and 'liquidations' in liquidation_data:
                df_liq = pd.DataFrame(liquidation_data['liquidations'])
                # Merge por timestamp
                dataframe = dataframe.merge(df_liq, on='date', how='left')
            return dataframe
        except Exception as e:
            logger.warning(f"Erro ao mesclar dados de liquidações: {e}")
            return dataframe

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
        return "CryptoHuntingStrategy v2.2 - Adaptive Anti-Apathy"

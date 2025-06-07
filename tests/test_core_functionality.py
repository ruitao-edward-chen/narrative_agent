"""
Unit tests for core functionality related to Narrative Agent and Transaction Model.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime, timedelta

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.narrative_agent import NarrativeAgentConfig
from src.narrative_agent.transaction_costs import TransactionCostModel
from src.narrative_agent.amm_pool import AMMPool
from src.narrative_agent.position_manager import PositionManager


class TestTransactionCostModel(unittest.TestCase):
    """
    Test the transaction cost model.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.model = TransactionCostModel(
            gas_fee_usd=50.0, amm_liquidity_usd=5_000_000.0, position_size_usd=50_000.0
        )

    def test_gas_fee_calculation(self):
        """
        Test gas fee is constant regardless of trade size.
        """
        costs1 = self.model.calculate_transaction_costs(10_000, 50_000, is_buy=True)
        costs2 = self.model.calculate_transaction_costs(50_000, 50_000, is_buy=True)

        self.assertEqual(costs1.gas_fee_usd, 50.0)
        self.assertEqual(costs2.gas_fee_usd, 50.0)

    def test_slippage_calculation(self):
        """
        Test slippage increases with trade size.
        """
        costs_small = self.model.calculate_transaction_costs(
            10_000, 50_000, is_buy=True
        )
        costs_large = self.model.calculate_transaction_costs(
            100_000, 50_000, is_buy=True
        )

        self.assertLess(costs_small.slippage_bps, costs_large.slippage_bps)
        self.assertLess(costs_small.slippage_usd, costs_large.slippage_usd)

    def test_round_trip_cost(self):
        """
        Test round trip cost calculation.
        """
        trade_size = 50_000
        entry_price = 50_000
        exit_price = 50_000
        position_type = 1

        rt_costs = self.model.calculate_round_trip_costs(
            trade_size, entry_price, exit_price, position_type
        )

        self.assertIn("entry_costs", rt_costs)
        self.assertIn("exit_costs", rt_costs)
        self.assertIn("total_cost_usd", rt_costs)
        self.assertIn("gross_pnl", rt_costs)

        expected_total = (
            rt_costs["entry_costs"].total_cost_usd
            + rt_costs["exit_costs"].total_cost_usd
        )
        self.assertAlmostEqual(rt_costs["total_cost_usd"], expected_total, places=2)


class TestAMMPool(unittest.TestCase):
    """
    Test the AMM pool simulation.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.pool = AMMPool(initial_liquidity_usd=1_000_000, fee_tier=0.003)
        # Sync to initial price
        self.pool.sync_to_price(50_000)

    def test_initialization(self):
        """
        Test pool initializes correctly.
        """
        self.assertEqual(self.pool.initial_liquidity, 1_000_000)
        # After syncing to price, check reserves
        self.assertIsNotNone(self.pool.reserve_base)
        self.assertIsNotNone(self.pool.reserve_quote)

    def test_buy_trade(self):
        """
        Test buy trades increase price.
        """
        # Get initial price
        initial_price = self.pool.reserve_quote / self.pool.reserve_base

        # Buy with 10K USD (USD -> base asset)
        amount_out, price_impact, effective_price = self.pool.calculate_swap_amounts(
            10_000, is_base_to_quote=False
        )

        # Calculate new price after trade
        new_reserve_quote = self.pool.reserve_quote + 10_000 * (1 - self.pool.fee_tier)
        new_reserve_base = self.pool.k / new_reserve_quote
        new_price = new_reserve_quote / new_reserve_base

        self.assertGreater(new_price, initial_price)
        self.assertGreater(price_impact, 0)

    def test_sell_trade(self):
        """
        Test sell trades decrease price.
        """
        # Get initial price
        initial_price = self.pool.reserve_quote / self.pool.reserve_base

        # Sell base asset worth 10K USD at current price
        base_amount = 10_000 / initial_price
        amount_out, price_impact, effective_price = self.pool.calculate_swap_amounts(
            base_amount, is_base_to_quote=True
        )

        # Calculate new price after trade
        new_reserve_base = self.pool.reserve_base + base_amount * (
            1 - self.pool.fee_tier
        )
        new_reserve_quote = self.pool.k / new_reserve_base
        new_price = new_reserve_quote / new_reserve_base

        self.assertLess(new_price, initial_price)
        self.assertGreater(price_impact, 0)

    def test_sync_to_external_price(self):
        """
        Test syncing pool to external price.
        """
        new_price = 55_000
        self.pool.sync_to_price(new_price)

        # Pool price should match external price
        pool_price = self.pool.reserve_quote / self.pool.reserve_base
        self.assertAlmostEqual(pool_price, new_price, places=2)

        # Reserves should still satisfy constant product
        k_after = self.pool.reserve_base * self.pool.reserve_quote
        self.assertAlmostEqual(k_after, self.pool.k, places=2)


class TestPositionManager(unittest.TestCase):
    """
    Test the position manager.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.manager = PositionManager(
            hold_period=1,
            transaction_cost=10,
            stop_loss=5.0,
            stop_gain=10.0,
        )

    def test_open_position(self):
        """
        Test opening a position.
        """
        narrative = {"ID": "test-narrative-123", "summary": "Test narrative summary"}
        entry_time = datetime.now().isoformat()

        self.manager.open_position(
            narrative=narrative, position=1, timestamp=entry_time, entry_price=50_000
        )

        self.assertIsNotNone(self.manager.active_position)
        position = self.manager.active_position
        self.assertEqual(position.narrative_id, "test-narrative-123")
        self.assertEqual(position.entry_price, 50_000)
        self.assertEqual(position.position, 1)

    def test_close_position_profit(self):
        """
        Test closing a profitable position.
        """
        narrative = {"ID": "test-narrative-123", "summary": "Test narrative summary"}
        entry_time = datetime.now().isoformat()

        self.manager.open_position(
            narrative=narrative, position=1, timestamp=entry_time, entry_price=50_000
        )

        exit_time = (datetime.now() + timedelta(minutes=30)).isoformat()
        self.manager.check_and_close_stop_conditions(exit_time, 55_000)

        self.assertIsNone(self.manager.active_position)
        self.assertEqual(len(self.manager.position_history), 1)
        closed = self.manager.position_history[0]
        self.assertEqual(closed.close_price, 55_000)
        self.assertGreater(closed.position_return, 0)
        self.assertEqual(closed.close_reason, "stop_gain")

    def test_close_position_loss(self):
        """
        Test closing a losing position.
        """
        narrative = {"ID": "test-narrative-123", "summary": "Test narrative summary"}
        entry_time = datetime.now().isoformat()

        self.manager.open_position(
            narrative=narrative,
            position=1,  # Long
            timestamp=entry_time,
            entry_price=50_000,
        )

        exit_time = (datetime.now() + timedelta(minutes=30)).isoformat()
        self.manager.check_and_close_stop_conditions(exit_time, 47_000)

        self.assertIsNone(self.manager.active_position)
        self.assertEqual(len(self.manager.position_history), 1)
        closed = self.manager.position_history[0]
        self.assertEqual(closed.close_price, 47_000)
        self.assertLess(closed.position_return, 0)
        self.assertEqual(closed.close_reason, "stop_loss")

    def test_hold_period_timeout(self):
        """
        Test position closes after hold period.
        """
        narrative = {"ID": "test-narrative-123", "summary": "Test narrative summary"}
        entry_time = datetime.now().isoformat()

        self.manager.open_position(
            narrative=narrative,
            position=1,  # Long
            timestamp=entry_time,
            entry_price=50_000,
        )

        # Move time past hold period
        exit_time = (datetime.now() + timedelta(hours=2)).isoformat()
        self.manager.check_and_close_expired_position(exit_time, 50_500)

        self.assertIsNone(self.manager.active_position)
        self.assertEqual(len(self.manager.position_history), 1)
        closed = self.manager.position_history[0]
        self.assertEqual(closed.close_reason, "hold_period_expired")


class TestNarrativeAgentConfig(unittest.TestCase):
    """
    Test the configuration class.
    """

    def test_default_config(self):
        """
        Test default configuration values.
        """
        config = NarrativeAgentConfig()

        self.assertEqual(config.ticker, "BTC")
        self.assertEqual(config.look_back_period, 6)
        self.assertEqual(config.hold_period, 1)
        self.assertTrue(config.use_enhanced_costs)

    def test_custom_config(self):
        """
        Test custom configuration.
        """
        config = NarrativeAgentConfig(
            ticker="ETH",
            look_back_period=12,
            hold_period=2,
            use_enhanced_costs=False,
            gas_fee_usd=100.0,
        )

        self.assertEqual(config.ticker, "ETH")
        self.assertEqual(config.look_back_period, 12)
        self.assertEqual(config.hold_period, 2)
        self.assertFalse(config.use_enhanced_costs)
        self.assertEqual(config.gas_fee_usd, 100.0)


if __name__ == "__main__":
    unittest.main()

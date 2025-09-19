"""
Fetch real-time Solana network data from various APIs.

This module provides functions to fetch current market and network data
to ensure simulations use the most up-to-date parameters.
"""

import json
import logging
import time
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class SolanaNetworkData:
    """Container for Solana network data."""
    sol_price_usd: float
    market_cap_usd: float
    total_supply: float
    circulating_supply: float
    total_staked_sol: float
    n_validators: int
    stake_concentration: float  # Gini coefficient or similar
    tvl_usd: float
    daily_volume_usd: float
    timestamp: datetime
    data_source: str


class DataFetcher:
    """Fetches real-time Solana network data from various sources."""
    
    # API endpoints
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    SOLANA_RPC = "https://api.mainnet-beta.solana.com"
    SOLANA_BEACH_API = "https://api.solanabeach.io/v1"
    VALIDATORS_APP_API = "https://www.validators.app/api/v1"
    
    # Cache settings
    CACHE_DURATION = timedelta(hours=1)  # Cache data for 1 hour
    CACHE_FILE = Path("data/network_data_cache.json")
    
    def __init__(self, use_cache: bool = True, timeout: int = 10):
        """
        Initialize data fetcher.
        
        Args:
            use_cache: Whether to use cached data if available
            timeout: API request timeout in seconds
        """
        self.use_cache = use_cache
        self.timeout = timeout
        self._cached_data: Optional[SolanaNetworkData] = None
        
        # Create SSL context that allows unverified certificates (for macOS)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Ensure cache directory exists
        if use_cache:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    def fetch_current_data(self) -> SolanaNetworkData:
        """
        Fetch current Solana network data from APIs.
        
        Returns fresh data or cached data if APIs fail.
        """
        # Check cache first
        if self.use_cache:
            cached = self._load_cache()
            if cached:
                logger.info(f"Using cached network data from {cached.timestamp}")
                return cached
        
        try:
            # Try to fetch fresh data
            data = self._fetch_from_apis()
            
            # Cache the data
            if self.use_cache:
                self._save_cache(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to fetch live data: {e}")
            
            # Try to use stale cache
            cached = self._load_cache(ignore_expiry=True)
            if cached:
                logger.warning("Using stale cached data as fallback")
                return cached
            
            # Return default values
            logger.warning("Using default values")
            return self._get_default_data()
    
    def _fetch_from_apis(self) -> SolanaNetworkData:
        """Fetch data from various APIs and combine results."""
        logger.info("Fetching live Solana network data...")
        
        # Get SOL price and market data from CoinGecko
        market_data = self._fetch_coingecko_data()
        
        # Get network stats from Solana RPC
        network_stats = self._fetch_solana_rpc_data()
        
        # Get validator data
        validator_data = self._fetch_validator_data()
        
        # Get DeFi TVL
        tvl_data = self._fetch_defi_tvl()
        
        # Combine all data
        return SolanaNetworkData(
            sol_price_usd=market_data.get('current_price', 235.0),
            market_cap_usd=market_data.get('market_cap', 100_000_000_000),
            total_supply=market_data.get('total_supply', 580_000_000),
            circulating_supply=market_data.get('circulating_supply', 420_000_000),
            total_staked_sol=network_stats.get('total_staked', 380_000_000),
            n_validators=validator_data.get('count', 1017),  # From Solana Beach
            stake_concentration=validator_data.get('gini_coefficient', 0.82),
            tvl_usd=tvl_data.get('tvl', 8_500_000_000),
            daily_volume_usd=market_data.get('total_volume', 3_800_000_000),
            timestamp=datetime.now(),
            data_source="Live APIs"
        )
    
    def _fetch_coingecko_data(self) -> Dict[str, Any]:
        """Fetch SOL market data from CoinGecko."""
        try:
            url = f"{self.COINGECKO_API}/coins/solana"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            full_url = f"{url}?{urlencode(params)}"
            
            with urllib.request.urlopen(full_url, timeout=self.timeout, context=self.ssl_context) as response:
                data = json.loads(response.read())
                
            market_data = data.get('market_data', {})
            
            return {
                'current_price': market_data.get('current_price', {}).get('usd', 235.0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 100_000_000_000),
                'total_volume': market_data.get('total_volume', {}).get('usd', 3_800_000_000),
                'circulating_supply': market_data.get('circulating_supply', 420_000_000),
                'total_supply': market_data.get('total_supply', 580_000_000)
            }
            
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"CoinGecko API error: {e}")
            return {}
    
    def _fetch_solana_rpc_data(self) -> Dict[str, Any]:
        """Fetch network stats from Solana RPC."""
        try:
            # Get total stake
            stake_data = self._make_rpc_request("getStakeActivation", {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getVoteAccounts"
            })
            
            if stake_data and 'result' in stake_data:
                current = stake_data['result'].get('current', [])
                total_stake = sum(v.get('activatedStake', 0) for v in current) / 1e9  # Convert lamports to SOL
                
                return {'total_staked': total_stake if total_stake > 0 else 380_000_000}
            
            return {}
            
        except Exception as e:
            logger.warning(f"Solana RPC error: {e}")
            return {}
    
    def _fetch_validator_data(self) -> Dict[str, Any]:
        """Fetch validator statistics."""
        try:
            # Try Solana RPC first
            vote_data = self._make_rpc_request("getVoteAccounts", {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getVoteAccounts"
            })
            
            if vote_data and 'result' in vote_data:
                current_validators = vote_data['result'].get('current', [])
                n_validators = len(current_validators)
                
                # Calculate Gini coefficient for stake distribution
                if current_validators:
                    stakes = sorted([v.get('activatedStake', 0) for v in current_validators])
                    gini = self._calculate_gini(stakes)
                else:
                    gini = 0.82
                
                return {
                    'count': n_validators if n_validators > 0 else 1017,  # From Solana Beach
                    'gini_coefficient': gini
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Validator data error: {e}")
            return {}
    
    def _fetch_defi_tvl(self) -> Dict[str, Any]:
        """Fetch DeFi TVL data."""
        try:
            # Try DeFiLlama API
            url = "https://api.llama.fi/tvl/solana"
            
            with urllib.request.urlopen(url, timeout=self.timeout, context=self.ssl_context) as response:
                tvl = float(response.read().decode())
                return {'tvl': tvl if tvl > 0 else 8_500_000_000}
            
        except Exception as e:
            logger.warning(f"DeFi TVL error: {e}")
            return {}
    
    def _make_rpc_request(self, method: str, payload: Dict) -> Optional[Dict]:
        """Make a request to Solana RPC."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = json.dumps(payload).encode()
            
            request = urllib.request.Request(
                self.SOLANA_RPC,
                data=data,
                headers=headers
            )
            
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                return json.loads(response.read())
                
        except Exception as e:
            logger.debug(f"RPC request failed: {e}")
            return None
    
    def _calculate_gini(self, values: list) -> float:
        """
        Calculate Gini coefficient for wealth distribution.
        
        0 = perfect equality, 1 = perfect inequality
        """
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = 0
        
        for i, value in enumerate(sorted_values):
            cumsum += (n - i) * value
        
        total = sum(sorted_values)
        if total == 0:
            return 0.0
        
        return (n + 1 - 2 * cumsum / total) / n
    
    def _load_cache(self, ignore_expiry: bool = False) -> Optional[SolanaNetworkData]:
        """Load cached data if available and not expired."""
        if not self.CACHE_FILE.exists():
            return None
        
        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.fromisoformat(data['timestamp'])
            
            # Check if cache is expired
            if not ignore_expiry and datetime.now() - timestamp > self.CACHE_DURATION:
                return None
            
            return SolanaNetworkData(
                sol_price_usd=data['sol_price_usd'],
                market_cap_usd=data['market_cap_usd'],
                total_supply=data['total_supply'],
                circulating_supply=data['circulating_supply'],
                total_staked_sol=data['total_staked_sol'],
                n_validators=data['n_validators'],
                stake_concentration=data['stake_concentration'],
                tvl_usd=data['tvl_usd'],
                daily_volume_usd=data['daily_volume_usd'],
                timestamp=timestamp,
                data_source=data.get('data_source', 'Cached')
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Cache load error: {e}")
            return None
    
    def _save_cache(self, data: SolanaNetworkData) -> None:
        """Save data to cache file."""
        try:
            cache_data = {
                'sol_price_usd': data.sol_price_usd,
                'market_cap_usd': data.market_cap_usd,
                'total_supply': data.total_supply,
                'circulating_supply': data.circulating_supply,
                'total_staked_sol': data.total_staked_sol,
                'n_validators': data.n_validators,
                'stake_concentration': data.stake_concentration,
                'tvl_usd': data.tvl_usd,
                'daily_volume_usd': data.daily_volume_usd,
                'timestamp': data.timestamp.isoformat(),
                'data_source': data.data_source
            }
            
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_default_data(self) -> SolanaNetworkData:
        """Return default data based on Dec 2024 values."""
        return SolanaNetworkData(
            sol_price_usd=235.0,
            market_cap_usd=110_000_000_000,
            total_supply=580_000_000,
            circulating_supply=470_000_000,
            total_staked_sol=380_000_000,
            n_validators=994,  # From Solana Beach (Sep 2025)
            stake_concentration=0.82,
            tvl_usd=12_740_000_000,  # From DefiLlama (Sep 2025)
            daily_volume_usd=3_800_000_000,
            timestamp=datetime.now(),
            data_source="Default values (Sep 2025)"
        )


def fetch_current_network_data(use_cache: bool = True) -> SolanaNetworkData:
    """
    Convenience function to fetch current network data.
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        Current Solana network data
    """
    fetcher = DataFetcher(use_cache=use_cache)
    return fetcher.fetch_current_data()


def update_config_with_live_data(config: Any, fetch_live: bool = True) -> None:
    """
    Update configuration with live network data.
    
    Args:
        config: Configuration object to update
        fetch_live: Whether to fetch live data or use defaults
    """
    if not fetch_live:
        logger.info("Using default network parameters")
        return
    
    try:
        logger.info("Fetching live network data...")
        data = fetch_current_network_data()
        
        # Update network parameters
        if hasattr(config, 'network'):
            config.network.n_validators = data.n_validators
            config.network.total_stake_sol = data.total_staked_sol
            config.network.stake_gini_coefficient = data.stake_concentration
            logger.info(f"Updated network params: {data.n_validators} validators, "
                       f"{data.total_staked_sol/1e6:.1f}M SOL staked")
        
        # Update economic parameters
        if hasattr(config, 'economic'):
            config.economic.sol_price_usd = data.sol_price_usd
            config.economic.total_value_locked_usd = data.tvl_usd
            config.economic.daily_volume_usd = data.daily_volume_usd
            logger.info(f"Updated economic params: SOL ${data.sol_price_usd:.2f}, "
                       f"TVL ${data.tvl_usd/1e9:.1f}B")
        
        # Log data source
        logger.info(f"Network data source: {data.data_source}")
        logger.info(f"Data timestamp: {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Failed to update config with live data: {e}")
        logger.info("Continuing with default values")


if __name__ == "__main__":
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Solana network data fetcher...")
    fetcher = DataFetcher(use_cache=False)  # Don't use cache for testing
    
    data = fetcher.fetch_current_data()
    
    print(f"\n{'='*50}")
    print("FETCHED SOLANA NETWORK DATA")
    print(f"{'='*50}")
    print(f"Data Source: {data.data_source}")
    print(f"Timestamp: {data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMarket Data:")
    print(f"  SOL Price: ${data.sol_price_usd:,.2f}")
    print(f"  Market Cap: ${data.market_cap_usd/1e9:,.1f}B")
    print(f"  Daily Volume: ${data.daily_volume_usd/1e9:,.1f}B")
    print(f"\nNetwork Data:")
    print(f"  Validators: {data.n_validators:,}")
    print(f"  Total Staked: {data.total_staked_sol/1e6:,.1f}M SOL")
    print(f"  Stake Concentration (Gini): {data.stake_concentration:.3f}")
    print(f"\nDeFi Data:")
    print(f"  Total Value Locked: ${data.tvl_usd/1e9:,.1f}B")
    print(f"{'='*50}")

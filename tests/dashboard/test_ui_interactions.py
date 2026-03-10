"""
Comprehensive UI Interaction Test for IDX Trading Dashboard.

Tests all pages and their API interactions to identify errors.
"""
import requests
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8502"

# Test results storage
results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "errors": []
}


def log_result(test_name: str, passed: bool, message: str = "", details: Any = None):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    entry = {
        "test": test_name,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }

    if passed:
        results["passed"].append(entry)
        print(f"{status}: {test_name}")
    else:
        results["failed"].append(entry)
        print(f"{status}: {test_name} - {message}")
        if details:
            print(f"    Details: {details}")


def log_warning(test_name: str, message: str):
    """Log a warning."""
    results["warnings"].append({"test": test_name, "message": message})
    print(f"⚠️  WARN: {test_name} - {message}")


def test_api_endpoint(method: str, endpoint: str, expected_status: int = 200,
                      json_data: Dict = None, test_name: str = None) -> Tuple[bool, Any]:
    """Test an API endpoint and return result."""
    url = f"{BASE_URL}{endpoint}"
    test_name = test_name or endpoint

    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        elif method == "POST":
            resp = requests.post(url, json=json_data, timeout=30)
        else:
            return False, "Invalid method"

        if resp.status_code == expected_status:
            try:
                data = resp.json()
                return True, data
            except:
                return True, resp.text[:200]
        else:
            return False, f"Status {resp.status_code}: {resp.text[:200]}"

    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - API not running?"
    except Exception as e:
        return False, str(e)


def test_page_apis():
    """Test all API endpoints used by each dashboard page."""

    print("\n" + "="*60)
    print("TESTING DASHBOARD PAGES API ENDPOINTS")
    print("="*60)

    # ============================================
    # HOME PAGE TESTS
    # ============================================
    print("\n--- HOME PAGE ---")

    success, data = test_api_endpoint("GET", "/health", test_name="Home: Health Check")
    log_result("Home: Health Check", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/health/detailed", test_name="Home: Detailed Health")
    log_result("Home: Detailed Health", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/health/data", test_name="Home: Data Freshness")
    log_result("Home: Data Freshness", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/health/update-status", test_name="Home: Update Status")
    log_result("Home: Update Status", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/health/dashboard-summary", test_name="Home: Dashboard Summary")
    log_result("Home: Dashboard Summary", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/stocks", test_name="Home: Stocks List")
    log_result("Home: Stocks List", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/signals", test_name="Home: Signals List")
    log_result("Home: Signals List", success, data if not success else "")

    # ============================================
    # SCREENER PAGE TESTS
    # ============================================
    print("\n--- SCREENER PAGE ---")

    success, data = test_api_endpoint("GET", "/stocks", test_name="Screener: Get Stocks")
    log_result("Screener: Get Stocks", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/health", test_name="Screener: Health Check")
    log_result("Screener: Health Check", success, data if not success else "")

    success, data = test_api_endpoint("POST", "/signals/scan", json_data={"symbols": ["BBCA"]}, test_name="Screener: Signal Scan")
    log_result("Screener: Signal Scan", success, data if not success else "")

    # ============================================
    # STOCK DETAIL PAGE TESTS
    # ============================================
    print("\n--- STOCK DETAIL PAGE ---")

    success, data = test_api_endpoint("GET", "/stocks/symbols", test_name="Stock Detail: Symbol List")
    log_result("Stock Detail: Symbol List", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/stocks/BBCA", test_name="Stock Detail: Get BBCA")
    log_result("Stock Detail: Get BBCA", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/stocks/BBCA/chart", test_name="Stock Detail: Chart Data")
    log_result("Stock Detail: Chart Data", success, data if not success else "")

    success, data = test_api_endpoint("POST", "/analysis/technical/BBCA", test_name="Stock Detail: Technical Analysis")
    log_result("Stock Detail: Technical Analysis", success, data if not success else "")

    success, data = test_api_endpoint("POST", "/fundamental/analyze", json_data={"symbol": "BBCA"}, test_name="Stock Detail: Fundamental Analysis")
    log_result("Stock Detail: Fundamental Analysis", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/stocks/BBCA/foreign-flow", test_name="Stock Detail: Foreign Flow")
    log_result("Stock Detail: Foreign Flow", success, data if not success else "")

    success, data = test_api_endpoint("POST", "/sentiment/fetch/BBCA", test_name="Stock Detail: Sentiment Fetch")
    log_result("Stock Detail: Sentiment Fetch", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/prediction/ensemble/BBCA", test_name="Stock Detail: ML Prediction")
    log_result("Stock Detail: ML Prediction", success, data if not success else "")

    # ============================================
    # SENTIMENT PAGE TESTS
    # ============================================
    print("\n--- SENTIMENT PAGE ---")

    success, data = test_api_endpoint("GET", "/sentiment/sector", test_name="Sentiment: Sector Data")
    log_result("Sentiment: Sector Data", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/sentiment/latest", test_name="Sentiment: Latest Articles")
    log_result("Sentiment: Latest Articles", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/sentiment/themes", test_name="Sentiment: Themes")
    log_result("Sentiment: Themes", success, data if not success else "")

    # ============================================
    # VIRTUAL TRADING PAGE TESTS
    # ============================================
    print("\n--- VIRTUAL TRADING PAGE ---")

    success, data = test_api_endpoint("GET", "/simulation/", test_name="Trading: List Sessions")
    log_result("Trading: List Sessions", success, data if not success else "")

    success, data = test_api_endpoint("POST", "/simulation/create", json_data={"initial_capital": 100000000}, test_name="Trading: Create Session")
    log_result("Trading: Create Session", success, data if not success else "")

    # ============================================
    # SETTINGS PAGE TESTS
    # ============================================
    print("\n--- SETTINGS PAGE ---")

    success, data = test_api_endpoint("GET", "/config/modes", test_name="Settings: Trading Modes")
    log_result("Settings: Trading Modes", success, data if not success else "")

    # ============================================
    # MARKET OVERVIEW PAGE TESTS
    # ============================================
    print("\n--- MARKET OVERVIEW PAGE ---")

    success, data = test_api_endpoint("GET", "/stocks", test_name="Market Overview: All Stocks")
    log_result("Market Overview: All Stocks", success, data if not success else "")

    # ============================================
    # ML PREDICTION PAGE TESTS
    # ============================================
    print("\n--- ML PREDICTION PAGE ---")

    success, data = test_api_endpoint("GET", "/prediction/ensemble/BBCA", test_name="ML: Ensemble Prediction")
    log_result("ML: Ensemble Prediction", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/prediction/correlation/BBCA", test_name="ML: Macro Correlation")
    log_result("ML: Macro Correlation", success, data if not success else "")

    success, data = test_api_endpoint("GET", "/prediction/commodities", test_name="ML: Commodities Data")
    log_result("ML: Commodities Data", success, data if not success else "")


def test_data_integrity():
    """Test data integrity and structure."""

    print("\n" + "="*60)
    print("TESTING DATA INTEGRITY")
    print("="*60)

    # Test stocks data structure
    success, data = test_api_endpoint("GET", "/stocks")
    if success:
        if isinstance(data, dict) and "stocks" in data:
            stocks = data["stocks"]
            if len(stocks) > 0:
                sample = stocks[0]
                required_fields = ["symbol", "name", "sector"]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    log_result("Data: Stock Structure", False, f"Missing fields: {missing}")
                else:
                    log_result("Data: Stock Structure", True)
            else:
                log_warning("Data: Stock Structure", "No stocks returned")
        elif isinstance(data, list):
            if len(data) > 0:
                sample = data[0]
                required_fields = ["symbol", "name", "sector"]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    log_result("Data: Stock Structure", False, f"Missing fields: {missing}")
                else:
                    log_result("Data: Stock Structure", True)
            else:
                log_warning("Data: Stock Structure", "Empty stocks list")
        else:
            log_result("Data: Stock Structure", False, f"Unexpected format: {type(data)}")

    # Test analysis data structure
    success, data = test_api_endpoint("GET", "/stocks/BBCA/analysis")
    if success:
        required_fields = ["symbol", "indicators"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            log_result("Data: Analysis Structure", False, f"Missing fields: {missing}")
        else:
            log_result("Data: Analysis Structure", True)


def test_error_handling():
    """Test error handling for invalid inputs."""

    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)

    # Test non-existent stock
    success, data = test_api_endpoint("GET", "/stocks/INVALID123")
    log_result("Error: Invalid Stock", not success, "Should return error for invalid stock")

    # Test invalid endpoint
    resp = requests.get(f"{BASE_URL}/invalid/endpoint", timeout=5)
    log_result("Error: 404 Handling", resp.status_code == 404, f"Got {resp.status_code}")


def generate_report():
    """Generate final test report."""

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results["passed"]) + len(results["failed"])
    pass_rate = (len(results["passed"]) / total * 100) if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    if results["failed"]:
        print("\n" + "-"*40)
        print("FAILED TESTS:")
        for fail in results["failed"]:
            print(f"  - {fail['test']}: {fail['message']}")

    if results["warnings"]:
        print("\n" + "-"*40)
        print("WARNINGS:")
        for warn in results["warnings"]:
            print(f"  - {warn['test']}: {warn['message']}")

    # Save report to file
    report_path = "/mnt/data/Project/idx-trading-system/tests/dashboard/test_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    return pass_rate >= 80


if __name__ == "__main__":
    print("IDX Trading Dashboard - UI Interaction Tests")
    print(f"API URL: {BASE_URL}")
    print(f"Dashboard URL: {DASHBOARD_URL}")
    print(f"Started: {datetime.now().isoformat()}")

    try:
        # Run all tests
        test_page_apis()
        test_data_integrity()
        test_error_handling()

        # Generate report
        success = generate_report()
        exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

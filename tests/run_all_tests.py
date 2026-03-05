"""
Master test runner — runs all test suites and prints a clear pass/fail summary.
"""
import unittest
import sys
import time

def run_all():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Load all test modules
    test_modules = [
        "test_irregular_detector",
        "test_condition_classifier",
        "test_adaptive_predictor",
    ]

    print("\n" + "="*60)
    print("   MENSTRUATION TRACKER — FULL TEST SUITE")
    print("="*60)

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
        except ModuleNotFoundError as e:
            print(f"❌ Could not load {module}: {e}")

    # Run with timing
    start = time.time()
    runner = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null","w"))
    result = runner.run(suite)
    elapsed = time.time() - start

    # Print summary
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print(f"\n{'─'*60}")
    print(f"  Results: {passed}/{total} tests passed in {elapsed:.2f}s")
    print(f"{'─'*60}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, msg in result.failures:
            print(f"  • {test}")
            print(f"    {msg.splitlines()[-1]}")

    if result.errors:
        print("\n💥 ERRORS:")
        for test, msg in result.errors:
            print(f"  • {test}")
            print(f"    {msg.splitlines()[-1]}")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {failed} test(s) need attention.")

    print("="*60 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
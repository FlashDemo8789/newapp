#!/usr/bin/env python3
"""
Integration Test Script for FlashDNA ML Modules

This script runs comprehensive integration tests to ensure that all ML modules
are properly connected between frontend and backend components.
"""

import os
import sys
import subprocess
import time
import argparse
import json
from datetime import datetime

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run FlashDNA ML integration tests')
parser.add_argument('--backend-only', action='store_true', help='Only test backend integration')
parser.add_argument('--frontend-only', action='store_true', help='Only test frontend integration')
parser.add_argument('--api-only', action='store_true', help='Only test API endpoints')
parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
parser.add_argument('--output', '-o', help='Write results to file')
args = parser.parse_args()

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")

def print_section(message):
    print(f"\n{Colors.BLUE}--- {message} ---{Colors.ENDC}\n")

def run_command(command, cwd=None, show_output=False):
    """Run a shell command and return (success, output)"""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd
    )
    
    stdout, stderr = process.communicate()
    output = stdout + stderr
    
    if show_output:
        print(output)
    
    return process.returncode == 0, output

def check_python_module_imports():
    """Check that all Python modules can be imported"""
    print_section("Checking Python Module Imports")
    
    # Run the module checker
    success, output = run_command(
        "python3 backend/adapters/ensure_modules.py",
        cwd=PROJECT_ROOT,
        show_output=args.verbose
    )
    
    if not success:
        print(f"{Colors.RED}❌ Module import check failed{Colors.ENDC}")
        return False
    
    # Count passed modules
    passed_count = output.count("✅ PASS")
    all_passed = "All modules available and ready!" in output
    
    if all_passed:
        print(f"{Colors.GREEN}✅ All modules passed import check ({passed_count} modules){Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠️ Some modules failed import check{Colors.ENDC}")
    
    if not args.verbose:
        print("Run with --verbose to see full output")
    
    return all_passed

def run_backend_integration_test():
    """Run backend integration tests"""
    print_section("Running Backend Integration Tests")
    
    success, output = run_command(
        "python3 backend/test_integration.py",
        cwd=PROJECT_ROOT,
        show_output=args.verbose
    )
    
    if not success:
        print(f"{Colors.RED}❌ Backend integration tests failed{Colors.ENDC}")
        return False
    
    # Count successful component tests
    passed_count = output.count("✅")
    error_count = output.count("❌")
    
    if error_count == 0:
        print(f"{Colors.GREEN}✅ All backend components integrated successfully ({passed_count} checks){Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠️ {error_count} backend integration checks failed{Colors.ENDC}")
    
    if not args.verbose:
        print("Run with --verbose to see full output")
    
    return error_count == 0

def is_api_running():
    """Check if the API is running"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('localhost', 5000))
        s.close()
        return True
    except:
        s.close()
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print_section("Testing API Endpoints")
    
    # First check if API is running
    if not is_api_running():
        print(f"{Colors.YELLOW}⚠️ API is not running on port 5000. Starting API server...{Colors.ENDC}")
        
        # Start API process
        api_process = subprocess.Popen(
            "python3 backend/app.py",
            shell=True,
            stdout=subprocess.PIPE if not args.verbose else None,
            stderr=subprocess.PIPE if not args.verbose else None,
            cwd=PROJECT_ROOT
        )
        
        # Wait for API to start
        print("Waiting for API to start...")
        attempts = 0
        while not is_api_running() and attempts < 10:
            time.sleep(1)
            attempts += 1
        
        if not is_api_running():
            print(f"{Colors.RED}❌ Failed to start API server{Colors.ENDC}")
            api_process.terminate()
            return False
        
        print(f"{Colors.GREEN}✅ API server started{Colors.ENDC}")
        api_needs_shutdown = True
    else:
        print(f"{Colors.GREEN}✅ API already running on port 5000{Colors.ENDC}")
        api_needs_shutdown = False
    
    # Run API tests
    success, output = run_command(
        "python3 backend/test_api.py",
        cwd=PROJECT_ROOT,
        show_output=args.verbose
    )
    
    # Shutdown API if we started it
    if api_needs_shutdown:
        print("Shutting down API server...")
        api_process.terminate()
    
    if not success:
        print(f"{Colors.RED}❌ API endpoint tests failed{Colors.ENDC}")
        return False
    
    # Count successful endpoint tests
    success_line = [line for line in output.split('\n') if "endpoints tested successfully" in line]
    if success_line:
        result = success_line[0].strip()
        print(f"{Colors.GREEN}✅ {result}{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠️ Could not determine endpoint test results{Colors.ENDC}")
    
    if not args.verbose:
        print("Run with --verbose to see full output")
    
    return success

def check_frontend_api_integration():
    """Test frontend API integration"""
    print_section("Checking Frontend API Integration")
    
    # Look at the frontend code to see if API calls are properly defined
    frontend_api_files = [
        os.path.join(FRONTEND_DIR, 'src/services/api.js'),
        os.path.join(FRONTEND_DIR, 'src/services/analysisService.js'),
        os.path.join(FRONTEND_DIR, 'src/directApi.js')
    ]
    
    missing_files = []
    for file_path in frontend_api_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"{Colors.RED}❌ Missing API integration files: {', '.join(missing_files)}{Colors.ENDC}")
        return False
    
    # Check if frontend is built
    if not os.path.exists(os.path.join(FRONTEND_DIR, 'build')):
        print(f"{Colors.YELLOW}⚠️ Frontend build directory not found, skipping build checks{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}✅ Frontend build exists{Colors.ENDC}")
    
    # Check for required API endpoints in the code
    required_endpoints = [
        "/analysis/monteCarlo",
        "/analysis/teamMoat",
        "/analysis/acquisition",
        "/analysis/pmf",
        "/analysis/technical",
        "/analysis/competitive",
        "/analysis/exitPath",
        "/analysis/cohort"
    ]
    
    api_file_contents = ""
    for file_path in frontend_api_files:
        try:
            with open(file_path, 'r') as f:
                api_file_contents += f.read()
        except Exception as e:
            print(f"{Colors.RED}❌ Error reading {file_path}: {str(e)}{Colors.ENDC}")
            return False
    
    missing_endpoints = []
    for endpoint in required_endpoints:
        if endpoint not in api_file_contents:
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        print(f"{Colors.RED}❌ Missing API endpoints in frontend code: {', '.join(missing_endpoints)}{Colors.ENDC}")
        return False
    
    print(f"{Colors.GREEN}✅ All required API endpoints found in frontend code{Colors.ENDC}")
    return True

def main():
    """Main function"""
    start_time = time.time()
    
    print_header("FlashDNA ML Integration Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "modules_import": None,
        "backend_integration": None,
        "api_endpoints": None,
        "frontend_integration": None
    }
    
    # Check module imports (always run this unless frontend-only)
    if not args.frontend_only:
        results["modules_import"] = check_python_module_imports()
    
    # Run backend integration tests
    if not args.frontend_only and not args.api_only:
        results["backend_integration"] = run_backend_integration_test()
    
    # Test API endpoints
    if not args.frontend_only and not args.backend_only:
        results["api_endpoints"] = test_api_endpoints()
    
    # Check frontend integration
    if not args.backend_only and not args.api_only:
        results["frontend_integration"] = check_frontend_api_integration()
    
    # Print summary
    print_header("Integration Test Summary")
    
    all_passed = True
    for test, result in results.items():
        if result is None:
            status = f"{Colors.BLUE}SKIPPED{Colors.ENDC}"
        elif result:
            status = f"{Colors.GREEN}PASSED{Colors.ENDC}"
        else:
            status = f"{Colors.RED}FAILED{Colors.ENDC}"
            all_passed = False
        
        # Format test name for display
        test_name = test.replace("_", " ").title()
        print(f"{test_name}: {status}")
    
    # Overall result
    print("\nOverall Result:", end=" ")
    if all_passed:
        print(f"{Colors.GREEN}ALL TESTS PASSED{Colors.ENDC}")
    else:
        print(f"{Colors.RED}SOME TESTS FAILED{Colors.ENDC}")
    
    # Show execution time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    
    # Write results to file if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {k: bool(v) if v is not None else None for k, v in results.items()},
            "overall_success": all_passed,
            "execution_time": elapsed_time
        }
        
        try:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults written to {args.output}")
        except Exception as e:
            print(f"\nError writing results to {args.output}: {str(e)}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 
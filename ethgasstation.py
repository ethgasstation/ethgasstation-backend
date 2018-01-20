#!/usr/bin/env python3
"""
    ETH Gas Station
    Primary backend.
"""

import argparse
from egs.main import master_control
from egs.output import Output

def main():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="An adaptive gas price oracle for Ethereum.")
    parser.add_argument('--generate-report', action='store_true', help="Generate reports for ethgasstation-frontend.")

    args = parser.parse_args()

    # kick off the egs main event loop
    master_control(args)

if __name__ == '__main__':
    o = Output()
    o.banner()
    main()

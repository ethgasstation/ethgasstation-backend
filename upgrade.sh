#!/bin/bash
set -e

# Run following command to execute the script:
# cd /usr/local/SettleFinance/ethgasstation-backend && git fetch --all && git reset --hard origin/master && git pull && chmod -R 777 /usr/local/SettleFinance/ethgasstation-backend/upgrade.sh && ./upgrade.sh

#For Initial Setup
#	mkdir -p -v /usr/local/SettleFinance
#	cd /usr/local/SettleFinance
#	git clone https://github.com/SettleFinance/ethgasstation-frontend.git
#	git clone https://github.com/SettleFinance/ethgasstation-backend.git
#	cd ethgasstation-backend

echo "####################################"
echo "# ETH GAS STARTION BACKEND UPGRADE #"
echo "####################################"

systemctl stop ethgassbackend

rm -v /usr/local/SettleFinance/ethgasstation-backend/settings.classic.conf || echo "Config file was probably already removed.";

cp -v /etc/ethgasstation/settings.conf /usr/local/SettleFinance/ethgasstation-backend/settings.conf

systemctl restart apache2.service
systemctl start ethgassbackend

#backend status verify
#journalctl --unit=ethgassbackend -n 100 --no-pager



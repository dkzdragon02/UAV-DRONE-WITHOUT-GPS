#!/bin/bash
# ============================================
# PX4 SITL Parameters cho Vision-Based OFFBOARD Flight
# ============================================
# Script này set tất cả PX4 params cần thiết cho VUAV-DRONE-WITHOUT-GPS
#
# CÁCH DÙNG:
#   Chạy trong PX4 shell (pxh>) sau khi PX4 SITL khởi động:
#     source ~/VUAV-DRONE-WITHOUT-GPS/scripts/px4_sitl_params.sh
#
#   Hoặc paste từng lệnh vào pxh> console
# ============================================

echo "============================================"
echo "THIẾT LẬP PX4 PARAMS CHO VISION OFFBOARD"
echo "============================================"
echo ""

cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║  PASTE CÁC LỆNH SAU VÀO PX4 SHELL (pxh>)                  ║
╚══════════════════════════════════════════════════════════════╝

# ========== EKF2: Vision-based, No GPS ==========
param set EKF2_EV_CTRL 15
param set EKF2_HGT_REF 3
param set EKF2_GPS_CTRL 0
param set EKF2_EV_DELAY 5

# ========== Disable Failsafes cho SITL ==========
# Battery: KHÔNG hành động khi pin thấp (SITL không cần)
param set COM_LOW_BAT_ACT 0

# RC Loss: KHÔNG hành động (OFFBOARD không cần RC)
param set NAV_RCL_ACT 0
param set COM_RCL_EXCEPT 4

# Data Link Loss: KHÔNG hành động
param set NAV_DLL_ACT 0

# Geofence: Tắt
param set GF_ACTION 0

# ========== OFFBOARD Config ==========
# Không auto-disarm khi chờ trên mặt đất
param set COM_DISARM_PRFLT -1

# OFFBOARD loss timeout (giây)
param set COM_OF_LOSS_T 5.0

# ========== Battery Thresholds (thấp để tránh trigger) ==========
param set BAT_LOW_THR 0.05
param set BAT_CRIT_THR 0.03
param set BAT_EMERGEN_THR 0.01

EOF

echo ""
echo "============================================"
echo "SAU KHI SET PARAMS, KHỞI ĐỘNG LẠI PX4:"
echo "  reboot"
echo "============================================"

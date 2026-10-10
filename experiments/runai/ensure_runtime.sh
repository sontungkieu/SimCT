#!/usr/bin/env bash
# ensure_runtime.sh — setup runtime idempotent cho mọi node, gọi 1 dòng:
#
#   bash /workspace/storage-shared/nlp/tungks/SimCT/ensure_runtime.sh
#
# Đã đủ (venv + python + portable-libs + nvrtc.h + torch cuda) thì in READY và
# thoát 0, không đụng gì. Thiếu thì verify SHA payload rồi giải nén đúng cây
# còn thiếu (không ghi đè cây có sẵn, không abort cả gói như install.sh cũ).
# Bản chạy thực tế là copy ở payload dir; sửa file này xong thì paste lại copy
# một lần (xem Block đặt file, không tự đồng bộ).
set -u
PAYLOAD=/workspace/storage-shared/nlp/tungks/SimCT
VENV_PY=/opt/venvs/simct-b200/bin/python
NVRTC=/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include/nvrtc.h

functional_ok() {
  [ -x "$VENV_PY" ] || return 1
  [ -n "$(ls -d /opt/python/cpython-3.12.12* 2>/dev/null)" ] || return 1
  [ -n "$(ls /opt/simct-portable-libs 2>/dev/null)" ] || return 1
  [ -f "$NVRTC" ] || return 1
  PYTHONWARNINGS=ignore "$VENV_PY" -c "import torch;assert torch.cuda.is_available()" 2>/dev/null
}

if functional_ok; then
  echo "RUNTIME READY (venv+python+libs+nvrtc+torch-cuda ok)"
  exit 0
fi
echo "thieu thanh phan -> cai dat selective"
cd "$PAYLOAD" || exit 2
sha256sum -c SHA256SUMS > /tmp/ensure_sums.log 2>&1 || {
  echo "VERIFY FAIL:"; tail -4 /tmp/ensure_sums.log; exit 3; }
echo "payload SHA ok"
[ -d /usr/local/cuda-13.0 ] || {
  echo "giai nen cuda-13.0 (dang thieu)"
  tar -xzf "$PAYLOAD/runtime.tar.gz" -C / usr/local/cuda-13.0 || exit 4; }
for d in opt/venvs opt/python opt/simct-portable-libs; do
  if [ -e "/$d" ]; then echo "giu nguyen /$d"
  else echo "giai nen $d"; tar -xzf "$PAYLOAD/runtime.tar.gz" -C / "$d" || exit 4; fi
done
if functional_ok; then
  echo "RUNTIME READY (vua cai xong)"
  exit 0
fi
echo "van hong sau selective -> xoa 3 cay cua ta va giai nen lai sach"
rm -rf /opt/venvs/simct-b200 /opt/simct-portable-libs
rm -rf /opt/python/cpython-3.12.12*
tar -xzf "$PAYLOAD/runtime.tar.gz" -C / opt/venvs opt/python opt/simct-portable-libs || exit 5
functional_ok && echo "RUNTIME READY (cai lai sach)" && exit 0
echo "RUNTIME FAIL - can can thiep tay"
exit 6

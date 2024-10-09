mkdir -p /app/.config/matplotlib
export MPLCONFIGDIR=/app/.config/matplotlib
rm -rf ~/.cache/matplotlib
font_path='/app/fonts/MALGUN.TTF'

# matplotlib에서 사용할 폰트 설정
echo "font.family: sans-serif" >> /app/.config/matplotlib/matplotlibrc
echo "font.sans-serif: Malgun Gothic" >> /app/.config/matplotlib/matplotlibrc
echo "axes.unicode_minus: False" >> /app/.config/matplotlib/matplotlibrc
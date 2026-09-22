@echo off
cd /d "%~dp0"
echo ====================================================
echo Starting chesSon Lichess Bot on @Harshiiittt...
echo ====================================================
python lichess_bot.py --token lip_63UwOWa79OR0uDvmCu5k --weights chesson\weights\chesson_weights.pt %*
pause

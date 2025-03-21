import os


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        # Verifica se il file esiste, altrimenti lo crea
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as file:
                file.write("Log file created.\n")

    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')

    def print_and_log(self, message):
        print(message)  # Mostra il messaggio a console
        self.log(message)  # Scrive il messaggio nel file di log

    def update_tp_sl(self, new_tp, new_sl):
        self.print_and_log(f"New TP: {new_tp}")
        self.print_and_log(f"New SL: {new_sl}")

    def print_check_TP(self, position_type, entry_price, current_price, take_profit):
        self.print_and_log("----Verifica TP POSIZIONE---")
        self.print_and_log(f"Action: {position_type}")
        self.print_and_log(f"Entry : {entry_price}")
        self.print_and_log(f"Price : {current_price}")
        msg = f"Take P : {entry_price * (1 + take_profit)} " if position_type == "long" else f"Take P : {entry_price * (1 - take_profit)} "
        self.print_and_log(msg)

    def print_check_SL(self, position_type, entry_price, current_price, stop_loss):
        self.print_and_log("----Verifica SL POSIZIONE---")
        self.print_and_log(f"Action: {position_type}")
        self.print_and_log(f"Entry : {entry_price}")
        self.print_and_log(f"Price : {current_price}")
        msg = f"SL: {entry_price * (1 - stop_loss)} " if position_type == "long" else f"SL : {entry_price * (1 + stop_loss)} "
        self.print_and_log(msg)

    def print_position_open(self, trade_action, entry_price):
        self.print_and_log("----APERTA POSIZIONE---")
        self.print_and_log(f"Action: {trade_action}")
        self.print_and_log(f"Entry : {entry_price}")

    def print_position_closed(self, trade_action, entry_price, close):
        self.print_and_log("----CHIUSA POSIZIONE---")
        self.print_and_log(f"Action: {trade_action}")
        self.print_and_log(f"Entry : {entry_price}")
        self.print_and_log(F"Close {close}")


# Utilizzo della classe
logger = Logger('logfile.txt')
logger.print_and_log("Questo è un messaggio di log.")

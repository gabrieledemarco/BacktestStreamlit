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


# Utilizzo della classe
logger = Logger('logfile.txt')
logger.print_and_log("Questo è un messaggio di log.")

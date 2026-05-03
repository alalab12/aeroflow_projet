"""
Module de communication réseau entre PC (master / slave)
Basé sur JSON + TCP, avec gestion correcte des messages.
"""

import socket
import json
import threading
from typing import Callable, Optional
import config


class NetworkServer:
    """
    Serveur réseau pour recevoir les données du PC esclave (côté MASTER)
    """

    def __init__(self, callback: Callable[[int], None]):
        self.callback = callback
        self.server_socket: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Démarre le serveur d'écoute"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("0.0.0.0", config.NETWORK_PORT))
            self.server_socket.listen(1)

            self.is_running = True
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()

            print(f"[SERVER] Démarré sur le port {config.NETWORK_PORT}")

        except Exception as e:
            print(f"[SERVER] Erreur serveur: {e}")

    def _listen(self):
        """Boucle d'acceptation + réception des données"""
        while self.is_running:
            try:
                print("[SERVER] En attente de connexion du slave...")
                # timeout pour pouvoir sortir proprement quand is_running passe à False
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                self.conn = conn
                print(f"[SERVER] Slave connecté depuis {addr}")

                buffer = ""
                while self.is_running:
                    chunk = conn.recv(config.BUFFER_SIZE)
                    if not chunk:
                        print("[SERVER] Connexion slave fermée")
                        break

                    buffer += chunk.decode("utf-8")

                    # Protocole: un JSON par ligne (terminé par '\n')
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            message = json.loads(line)
                            count = int(message.get("count", 0))
                            print(f"[SERVER] Reçu du slave: {count}")
                            # Appel du callback => on_remote_count sur le master
                            self.callback(count)
                        except Exception as e:
                            print(f"[SERVER] Erreur décodage JSON: {e} | line={line}")

                try:
                    conn.close()
                except Exception:
                    pass
                self.conn = None

            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[SERVER] Erreur réception: {e}")

    def stop(self):
        """Arrête le serveur"""
        self.is_running = False
        try:
            if self.conn:
                self.conn.close()
            if self.server_socket:
                self.server_socket.close()
        except Exception:
            pass
        print("[SERVER] Arrêté")


class NetworkClient:
    """
    Client réseau pour envoyer les données au PC maître (côté SLAVE)
    Connexion persistante tant que l'esclave tourne.
    """

    def __init__(self, master_ip: str):
        self.master_ip = master_ip
        self.sock: Optional[socket.socket] = None
        print(f"[CLIENT] Initialisation NetworkClient vers {self.master_ip}")
        self._connect()

    def _connect(self):
        """Établit la connexion au maître (avec retry simple)"""
        while self.sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.master_ip, config.NETWORK_PORT))
                s.settimeout(None)  # mode bloquant normal après la connexion
                self.sock = s
                print(f"[CLIENT] Connecté au maître {self.master_ip}:{config.NETWORK_PORT}")
            except Exception as e:
                print(f"[CLIENT] Échec connexion maître: {e}, nouvelle tentative...")
                try:
                    s.close()
                except Exception:
                    pass

    def send_count(self, count: int) -> bool:
        """
        Envoie le comptage au PC maître.
        Retourne True si succès, False sinon.
        """
        if self.sock is None:
            self._connect()

        try:
            message = json.dumps({"count": int(count)}) + "\n"
            self.sock.sendall(message.encode("utf-8"))
            print(f"[CLIENT] send_count -> {count}")
            return True

        except Exception as e:
            print(f"[CLIENT] Erreur envoi réseau: {e}")
            # La connexion a été coupée (10053, etc.) -> on la refera au prochain appel
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            return False

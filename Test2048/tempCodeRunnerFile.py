 (n_games, n_games / 1000)
                + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {}".format(
                    mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile
                ),
            )
    except KeyboardInterrupt:
        print("training interrupted")
        print("{} games played by the agent".format(n_games))
        if input("save the agent? (y/n)") == "y":
            fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
            pickle.dump((n_games, agent), open(fout, "wb"))
            print("agent saved to", fout)
            
            
    print("{} games played by the agent".format(n_games))
    if input("save the agent? (y/n)") == "y":
        fout = "E:/myProgrammings/Ai/ML/2048-RL/data/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
        pickle.dump((n_games, agent), open(fout, "wb"))
        print("agent saved to", fout)
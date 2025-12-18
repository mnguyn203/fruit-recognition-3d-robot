from robot_system import RobotVisionSystem


def main():
    print("🤖 ROBOT VISION SYSTEM (CLEAN OUTPUT)")
    print("=" * 40)
    print("Press 'D' to toggle debug mode anytime")

    try:
        # Start in quiet mode (debug=False)
        system = RobotVisionSystem(debug=False)
        system.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        print("👋 System shutdown")


if __name__ == "__main__":
    main()
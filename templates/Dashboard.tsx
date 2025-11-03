import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { api } from '../lib/api';
import { TrendingUp, Target, MessageCircle, UtensilsCrossed, History, Send } from 'lucide-react';

interface DashboardProps {
  onNavigateTo: (page: string) => void;
}

export function Dashboard({ onNavigateTo }: DashboardProps) {
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    const data = await api.getDashboard();
    setDashboardData(data);
    setLoading(false);
  };

  if (loading || !dashboardData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-red-50 p-4 md:p-8 flex items-center justify-center">
        <div className="text-green-700">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-red-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-green-700 mb-6">Dashboard</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card className="border-2 border-green-200">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-green-700">Week Progress</CardTitle>
              <TrendingUp className="w-4 h-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl mb-2">
                {dashboardData.weekProgress.completed}/{dashboardData.weekProgress.total}
              </div>
              <Progress value={dashboardData.weekProgress.percentage} className="h-2 mb-2" />
              <p className="text-sm text-gray-600">{dashboardData.weekProgress.percentage}% complete</p>
            </CardContent>
          </Card>

          <Card className="border-2 border-green-200">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-green-700">Target</CardTitle>
              <Target className="w-4 h-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl mb-2">
                {dashboardData.target.current}/{dashboardData.target.goal}
              </div>
              <Progress value={(dashboardData.target.current / dashboardData.target.goal) * 100} className="h-2 mb-2" />
              <p className="text-sm text-gray-600">{dashboardData.target.unit} this month</p>
            </CardContent>
          </Card>

          <Card 
            className="border-2 border-green-200 cursor-pointer hover:shadow-lg transition-all"
            onClick={() => onNavigateTo('history')}
          >
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-green-700">History</CardTitle>
              <History className="w-4 h-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl mb-2">{dashboardData.recentWorkouts}</div>
              <p className="text-sm text-gray-600">Recent workouts</p>
              <Button variant="link" className="p-0 h-auto text-green-600 text-sm mt-2">
                View all →
              </Button>
            </CardContent>
          </Card>

          <Card className="border-2 border-green-200">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm text-green-700">Diet</CardTitle>
              <UtensilsCrossed className="w-4 h-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl mb-2">{dashboardData.diet.calories}</div>
              <p className="text-sm text-gray-600">kcal today</p>
              <div className="flex gap-2 mt-2">
                <Badge variant="outline" className="text-xs">P: {dashboardData.diet.protein}g</Badge>
                <Badge variant="outline" className="text-xs">C: {dashboardData.diet.carbs}g</Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card 
            className="border-2 border-green-200 cursor-pointer hover:shadow-lg transition-all"
            onClick={() => onNavigateTo('chatbot')}
          >
            <CardHeader>
              <CardTitle className="text-green-700 flex items-center gap-2">
                <MessageCircle className="w-5 h-5" />
                AI Fitness Coach
              </CardTitle>
              <CardDescription>Get personalized fitness advice</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="h-48 flex items-center justify-center bg-gradient-to-br from-green-50 to-red-50 rounded-lg border-2 border-green-200">
                  <div className="text-center">
                    <MessageCircle className="w-16 h-16 text-green-600 mx-auto mb-3" />
                    <p className="text-sm text-gray-600 mb-2">Chat with your AI coach</p>
                    <Button className="bg-green-600 hover:bg-green-700 mt-2">
                      Open Chat →
                    </Button>
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  <p>Get help with:</p>
                  <ul className="mt-2 space-y-1">
                    <li className="flex items-center gap-2">
                      <span className="text-green-600">•</span>
                      <span>Workout recommendations</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-green-600">•</span>
                      <span>Nutrition advice</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-green-600">•</span>
                      <span>Form corrections</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-2 border-green-200">
            <CardHeader>
              <CardTitle className="text-green-700">Nutrition Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Protein</span>
                  <span>{dashboardData.diet.protein}g / 150g</span>
                </div>
                <Progress value={(dashboardData.diet.protein / 150) * 100} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Carbohydrates</span>
                  <span>{dashboardData.diet.carbs}g / 250g</span>
                </div>
                <Progress value={(dashboardData.diet.carbs / 250) * 100} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Fats</span>
                  <span>{dashboardData.diet.fats}g / 70g</span>
                </div>
                <Progress value={(dashboardData.diet.fats / 70) * 100} className="h-2" />
              </div>
              
              <div className="pt-4 border-t border-green-200">
                <h4 className="text-sm mb-3">Quick Tips</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-start gap-2">
                    <span className="text-green-600">•</span>
                    <span>Stay hydrated - drink 8 glasses of water daily</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-600">•</span>
                    <span>Eat protein within 30 minutes post-workout</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-600">•</span>
                    <span>Include vegetables in every meal</span>
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
